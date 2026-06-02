# Whale-watching fleet — Xcode / Claude brief
### Standalone spec for an iOS app. Pure API documentation — no Python dependencies.

You're building a **native iOS app** that fetches the live positions of a
curated Vancouver-area whale-watching fleet (17 vessels) from public AIS
data sources and renders them on a map. This document is the complete
brief: data sources, HTTP/WebSocket specs, Swift code patterns, the full
fleet table with MMSIs, and rendering guidance.

Target stack: **Swift 5.9+, iOS 17+, SwiftUI, async/await, MapKit**.
Networking via `URLSession` only — no third-party deps required.

---

## 1. What you're building

Given the fleet table in §5, periodically resolve each vessel's current
**MMSI → position → speed → heading → last-update time**, then render
them as colored map annotations with a sheet showing details on tap.

The user-facing shape (one record per vessel that has a recent fix):

```swift
struct VesselFix: Identifiable, Hashable {
    let id: Int            // MMSI
    let name: String       // human name, e.g. "Salish Sea Freedom"
    let operatorName: String   // "Prince of Whales", etc.
    let coordinate: CLLocationCoordinate2D
    let speedKnots: Double?    // sustained speed over ground
    let courseDegrees: Double? // 0..360, true bearing of motion
    let courseCompass: String  // e.g. "WNW"
    let lastUpdateUTC: Date    // when the AIS report was emitted
    var ageMinutes: Int { Int(Date().timeIntervalSince(lastUpdateUTC) / 60) }
    var isStale: Bool { ageMinutes > 30 }
    let iconColor: Color   // per-operator (see §5)
}
```

**Stale rule**: AIS broadcasts every 2–10 s when underway. Anything older
than 30 min means moored, AIS off, or out of receiver range — flag but
don't hide.

---

## 2. Pick a data source

There are two viable public sources. Use one as primary; both are
optional. **For an iOS app, VesselAPI REST is recommended** (simpler,
no long-lived connections fighting iOS background limits).

| | **VesselAPI REST (A)** | **AISStream WebSocket (B)** |
|---|---|---|
| Style | Per-vessel polling | Push, area subscription |
| Auth | `x-api-key` header | API key in subscribe payload |
| iOS fit | Excellent (HTTP/2, easy) | Awkward (background drops connection) |
| Rate | Tier-based, ~250/day free | Open in bbox, free |
| Latency | Whatever you poll at | Real-time (sub-second) |
| Code | URLSession dataTask | URLSessionWebSocketTask + Task |

If you want both: poll REST every 60 s as the default; open WS only on
the foreground tracking screen for live updates.

---

## 3. Source A — VesselAPI Ship Tracking (REST)

### 3.1 Base + auth

```
Base URL:  https://api.vesselapi.com
Header:    x-api-key: <YOUR_KEY>
```

Store the key in **Keychain**, not `Info.plist`. Never bundle it in the
shipped binary — proxy it through your own server if the app will be
distributed.

### 3.2 Endpoints

#### Current position for one vessel (primary)
```
GET /vessel/{mmsi}/position?filter.idType=mmsi
```

**Success (200) response** — note the camelCase wrapper key:
```json
{
  "vesselPosition": {
    "mmsi": 316042213,
    "shipName": "SALISH SEA FREEDOM",
    "latitude": 49.286,
    "longitude": -123.118,
    "speed": 18.4,
    "course": 285,
    "heading": 287,
    "timestamp": "2026-05-29T01:17:16Z",
    "destination": "VANCOUVER",
    "navStatus": "Under way using engine"
  }
}
```

**Field semantics:**
- `speed` — knots, SOG (speed over ground). Can be 0–102 (102 = "not available").
- `course` — degrees true, COG (course over ground). Use this for the
  on-screen arrow direction.
- `heading` — degrees true, bow direction. Often equal to course but
  can differ at low speed / when drifting. Prefer `course` for the arrow.
- `timestamp` — ISO 8601, **UTC, `Z` suffix guaranteed**. Decode with
  `ISO8601DateFormatter`.
- `shipName` — uppercase, no punctuation. Don't rely on for matching;
  prefer the MMSI you queried.

**404 response** means "no recent AIS fix" — vessel dockside, AIS off,
or out of range. **This is normal**, not an error to log loudly.

**429 response** means rate-limited — back off exponentially, optionally
fall through to a backup key (see §3.3).

#### Resolve MMSI from a name (only if MMSI unknown)
```
GET /vessel/search?filter.name=Salish%20Sea%20Freedom
```
Returns candidates. Cache the resulting MMSI **permanently** — name→MMSI
is stable per hull. The fleet table in §5 already pre-supplies the
MMSIs we know; only call this for vessels where MMSI is `—`.

### 3.3 Swift client (drop-in)

```swift
import Foundation
import CoreLocation

actor VesselAPIClient {
    private let primaryKey: String
    private let backupKey: String?
    private let session: URLSession

    init(primaryKey: String, backupKey: String? = nil) {
        self.primaryKey = primaryKey
        self.backupKey = backupKey
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 12
        cfg.httpMaximumConnectionsPerHost = 8
        self.session = URLSession(configuration: cfg)
    }

    private struct Envelope: Decodable { let vesselPosition: PositionDTO? }
    private struct PositionDTO: Decodable {
        let mmsi: Int
        let shipName: String?
        let latitude: Double
        let longitude: Double
        let speed: Double?
        let course: Double?
        let heading: Double?
        let timestamp: String   // ISO 8601 Z
    }

    /// Returns a fix or nil (404 / no AIS data).
    func position(mmsi: Int) async throws -> VesselFix? {
        return try await positionTry(mmsi: mmsi, key: primaryKey)
            ?? (backupKey.flatMap { try? await positionTry(mmsi: mmsi, key: $0) } ?? nil)
    }

    private func positionTry(mmsi: Int, key: String) async throws -> VesselFix? {
        var req = URLRequest(url: URL(string:
            "https://api.vesselapi.com/vessel/\(mmsi)/position?filter.idType=mmsi")!)
        req.setValue(key, forHTTPHeaderField: "x-api-key")
        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse else { return nil }
        if http.statusCode == 404 { return nil }                // no fix
        if http.statusCode == 429 { return nil }                // rate limit → caller falls back
        guard (200..<300).contains(http.statusCode) else { return nil }

        let envelope = try JSONDecoder().decode(Envelope.self, from: data)
        guard let p = envelope.vesselPosition else { return nil }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let ts = iso.date(from: p.timestamp)
            ?? ISO8601DateFormatter().date(from: p.timestamp)
            ?? Date()

        // Look up fleet metadata (operator + color) by MMSI.
        let meta = WhaleFleet.byMMSI[mmsi]
        return VesselFix(
            id: p.mmsi,
            name: meta?.name ?? (p.shipName?.capitalized ?? "Unknown"),
            operatorName: meta?.operatorName ?? "Unknown",
            coordinate: .init(latitude: p.latitude, longitude: p.longitude),
            speedKnots: p.speed,
            courseDegrees: p.course,
            courseCompass: degToCompass(p.course) ?? "—",
            lastUpdateUTC: ts,
            iconColor: meta?.color ?? .gray
        )
    }
}
```

### 3.4 Polling strategy

- **Default: 60 s per vessel**. Cache the last fix and only re-render
  when coordinates change ≥ 50 m or > 30 s elapsed.
- Don't run while the app is backgrounded. Re-poll on `scenePhase ==
  .active`. iOS will throttle/kill long-running background timers anyway.
- Hit `/vessel/{mmsi}/position` concurrently (up to ~8 parallel tasks);
  don't serialize a 17-vessel sweep — it'd take 17 × RTT.

```swift
let fleet = WhaleFleet.all
let fixes: [VesselFix] = try await withThrowingTaskGroup(of: VesselFix?.self) { group in
    for v in fleet { group.addTask { try? await client.position(mmsi: v.mmsi) } }
    var out: [VesselFix] = []
    for try await fix in group { if let f = fix { out.append(f) } }
    return out
}
```

---

## 4. Source B — AISStream.io (WebSocket)

Use only when you really want sub-second updates on the foreground
tracking screen. Reconnect on disconnect; cancel when the app backgrounds.

### 4.1 Connect + subscribe

```
URL: wss://stream.aisstream.io/v0/stream
```

Send this **as the very first frame** after connect:

```json
{
  "APIKey": "<YOUR_AISSTREAM_KEY>",
  "BoundingBoxes": [[[48.6, -124.4], [49.9, -122.7]]],
  "FilterMessageTypes": ["PositionReport", "ShipStaticData"]
}
```

The bbox `[[lat_sw, lon_sw], [lat_ne, lon_ne]]` covers
**Strait of Georgia + Howe Sound + Burrard Inlet + south to the US border**.

### 4.2 Messages you'll receive

**PositionReport** (every 2–10 s while underway):
```json
{
  "MessageType": "PositionReport",
  "MetaData": {
    "MMSI": 316042213,
    "ShipName": "SALISH SEA FREEDOM",
    "time_utc": "2026-05-29 01:17:16 +0000 UTC",
    "latitude": 49.286,
    "longitude": -123.118
  },
  "Message": {
    "PositionReport": {
      "Sog": 18.4,            // speed over ground, knots
      "Cog": 285.0,           // course over ground, degrees
      "TrueHeading": 287,     // bow heading, degrees
      "Latitude": 49.286,
      "Longitude": -123.118,
      "NavigationalStatus": 0
    }
  }
}
```

**ShipStaticData** (every ~6 min — needed only to learn `MMSI → name`):
```json
{
  "MessageType": "ShipStaticData",
  "MetaData": { "MMSI": 316042213, "ShipName": "SALISH SEA FREEDOM" },
  "Message": { "ShipStaticData": { "Name": "SALISH SEA FREEDOM" } }
}
```

Once you have an MMSI → name mapping, **cache it permanently** (vessels
keep their MMSI for life). You don't need ShipStaticData on subsequent
runs if the fleet is in §5.

### 4.3 Swift WebSocket client (sketch)

```swift
import Foundation

final class AISStreamClient {
    private var task: URLSessionWebSocketTask?
    private let url = URL(string: "wss://stream.aisstream.io/v0/stream")!
    private let apiKey: String
    private(set) var onFix: (VesselFix) -> Void = { _ in }

    init(apiKey: String) { self.apiKey = apiKey }

    func start(filterMMSIs: Set<Int>, fixHandler: @escaping (VesselFix) -> Void) {
        onFix = fixHandler
        task = URLSession.shared.webSocketTask(with: url)
        task?.resume()

        let sub: [String: Any] = [
            "APIKey": apiKey,
            "BoundingBoxes": [[[48.6, -124.4], [49.9, -122.7]]],
            "FilterMessageTypes": ["PositionReport"]
        ]
        let subData = try! JSONSerialization.data(withJSONObject: sub)
        task?.send(.string(String(data: subData, encoding: .utf8)!)) { _ in }
        Task { await self.receiveLoop(filter: filterMMSIs) }
    }

    func stop() { task?.cancel(with: .goingAway, reason: nil); task = nil }

    private func receiveLoop(filter: Set<Int>) async {
        guard let task else { return }
        do {
            while true {
                let msg = try await task.receive()
                guard case .string(let s) = msg, let data = s.data(using: .utf8),
                      let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
                else { continue }
                guard let meta = dict["MetaData"] as? [String: Any],
                      let mmsi = meta["MMSI"] as? Int, filter.contains(mmsi)
                else { continue }
                if let fix = parsePositionReport(dict: dict, mmsi: mmsi) {
                    await MainActor.run { onFix(fix) }
                }
            }
        } catch {
            // reconnect after a backoff
            try? await Task.sleep(for: .seconds(3))
            start(filterMMSIs: filter, fixHandler: onFix)
        }
    }

    private func parsePositionReport(dict: [String: Any], mmsi: Int) -> VesselFix? {
        guard let message = dict["Message"] as? [String: Any],
              let pos = message["PositionReport"] as? [String: Any],
              let lat = (pos["Latitude"] ?? pos["latitude"]) as? Double,
              let lon = (pos["Longitude"] ?? pos["longitude"]) as? Double
        else { return nil }
        let sog = pos["Sog"] as? Double
        let cog = pos["Cog"] as? Double
        let meta = WhaleFleet.byMMSI[mmsi]
        let timeStr = (dict["MetaData"] as? [String: Any])?["time_utc"] as? String
        let ts = parseAISTimestamp(timeStr) ?? Date()
        return VesselFix(
            id: mmsi,
            name: meta?.name ?? "Unknown",
            operatorName: meta?.operatorName ?? "Unknown",
            coordinate: .init(latitude: lat, longitude: lon),
            speedKnots: sog, courseDegrees: cog,
            courseCompass: degToCompass(cog) ?? "—",
            lastUpdateUTC: ts,
            iconColor: meta?.color ?? .gray
        )
    }
}

func parseAISTimestamp(_ s: String?) -> Date? {
    guard let s else { return nil }
    let fmt = DateFormatter()
    fmt.locale = .init(identifier: "en_US_POSIX")
    fmt.dateFormat = "yyyy-MM-dd HH:mm:ss Z 'UTC'"
    fmt.timeZone = .init(secondsFromGMT: 0)
    return fmt.date(from: s)
}
```

iOS reality: when the app backgrounds, the WebSocket dies. Call `stop()`
on `scenePhase != .active` and `start(...)` on return; the foreground
state is the only one with a reliable live connection.

---

## 5. Fleet (17 vessels)

Define this once as a Swift constant. Match incoming records **by
MMSI** when possible.

```swift
import SwiftUI

struct FleetMember {
    let mmsi: Int           // 0 if unknown
    let name: String
    let operatorName: String
    let color: Color
}

enum WhaleFleet {
    static let all: [FleetMember] = [
        // Wild Whales Vancouver — blue
        .init(mmsi: 316040487, name: "Aurora I",            operatorName: "Wild Whales Vancouver",  color: .blue),
        .init(mmsi: 316040366, name: "Aurora II",           operatorName: "Wild Whales Vancouver",  color: .blue),
        .init(mmsi: 316034894, name: "Eagle Eyes",          operatorName: "Wild Whales Vancouver",  color: .blue),
        .init(mmsi: 316032442, name: "Jing Yu",             operatorName: "Wild Whales Vancouver",  color: .blue),

        // Vancouver Whale Watch — green
        .init(mmsi: 316008046, name: "Explorathor II",      operatorName: "Vancouver Whale Watch",  color: .green),
        .init(mmsi: 316008045, name: "Explorathor Express", operatorName: "Vancouver Whale Watch",  color: .green),
        .init(mmsi: 0,         name: "Express",             operatorName: "Vancouver Whale Watch",  color: .green),
        .init(mmsi: 316035167, name: "Strider",             operatorName: "Vancouver Whale Watch",  color: .green),
        .init(mmsi: 316014609, name: "Lightship",           operatorName: "Vancouver Whale Watch",  color: .green),

        // Prince of Whales — orange
        .init(mmsi: 316032858, name: "Salish Sea Dream",    operatorName: "Prince of Whales",       color: .orange),
        .init(mmsi: 316042213, name: "Salish Sea Freedom",  operatorName: "Prince of Whales",       color: .orange),
        .init(mmsi: 316039686, name: "Salish Sea Eclipse",  operatorName: "Prince of Whales",       color: .orange),
        .init(mmsi: 316059231, name: "Salish Sea Glory",    operatorName: "Prince of Whales",       color: .orange),
        .init(mmsi: 316006789, name: "Ocean Magic",         operatorName: "Prince of Whales",       color: .orange),
        .init(mmsi: 316008331, name: "Ocean Magic II",      operatorName: "Prince of Whales",       color: .orange),

        // Other / private — purple (consecutive MMSIs suggest sister vessels)
        .init(mmsi: 316004454, name: "The Duchess",         operatorName: "Other",                  color: .purple),
        .init(mmsi: 316004455, name: "Countess",            operatorName: "Other",                  color: .purple),
        .init(mmsi: 316004456, name: "Lady Di",             operatorName: "Other",                  color: .purple),
    ]

    static var knownMMSIs: Set<Int> { Set(all.compactMap { $0.mmsi > 0 ? $0.mmsi : nil }) }
    static var byMMSI: [Int: FleetMember] { Dictionary(uniqueKeysWithValues: all.compactMap { $0.mmsi > 0 ? ($0.mmsi, $0) : nil }) }
}
```

**Only "Express"** is still without an MMSI — call
`/vessel/search?filter.name=Express` once you observe it on AIS, copy
the result into the table, and skip the runtime search forever after.

### MMSIs at a glance (16 of 17 known)

```
316004454  The Duchess
316004455  Countess
316004456  Lady Di
316006789  Ocean Magic
316008045  Explorathor Express
316008046  Explorathor II
316008331  Ocean Magic II
316014609  Lightship
316032442  Jing Yu
316032858  Salish Sea Dream
316034894  Eagle Eyes
316035167  Strider
316039686  Salish Sea Eclipse
316040366  Aurora II
316040487  Aurora I
316042213  Salish Sea Freedom
316059231  Salish Sea Glory
```

---

## 6. Compass helper (drop-in)

```swift
private let compassLabels = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                             "S","SSW","SW","WSW","W","WNW","NW","NNW"]

func degToCompass(_ deg: Double?) -> String? {
    guard let deg, deg.isFinite else { return nil }
    let bucket = Int(((deg.truncatingRemainder(dividingBy: 360) + 360)
                      .truncatingRemainder(dividingBy: 360) / 22.5).rounded()) % 16
    return compassLabels[bucket]
}
```

---

## 7. Map rendering (MapKit + SwiftUI)

iOS 17+ uses the new `Map` API. One marker per vessel, colored per
operator, opacity reduced when stale.

```swift
import SwiftUI
import MapKit

struct FleetMapView: View {
    @StateObject var vm: FleetViewModel  // owns the [VesselFix]

    var body: some View {
        Map {
            ForEach(vm.fixes) { fix in
                Annotation(fix.name, coordinate: fix.coordinate) {
                    VStack(spacing: 2) {
                        Image(systemName: "ferry.fill")
                            .foregroundStyle(fix.iconColor)
                            .opacity(fix.isStale ? 0.4 : 1.0)
                            .rotationEffect(.degrees(fix.courseDegrees ?? 0))
                        if let sog = fix.speedKnots {
                            Text("\(Int(sog)) kt")
                                .font(.caption2.bold())
                                .padding(.horizontal, 4)
                                .background(.white.opacity(0.85), in: .capsule)
                        }
                    }
                }
            }
        }
        .task { await vm.start() }
    }
}
```

The view model holds the `actor VesselAPIClient` and runs a periodic
`Task { while !Task.isCancelled { … sleep(60) } }` to refresh.

---

## 8. Caching & rate limiting

- **MMSI cache**: `UserDefaults` or a small JSON file in
  `applicationSupportDirectory`. Once learned, MMSIs are permanent.
- **Position cache**: in-memory dictionary `[MMSI: VesselFix]` so
  intermittent 404s don't make markers blink off the map immediately —
  keep showing the last fix with `isStale = true` until you can confirm
  the vessel is gone.
- **Backoff on 429**: exponential, max 5 min. If both primary and
  backup keys are 429ing, suspend polling for 5 min and show a banner.
- **Don't poll faster than 60 s/vessel** on the free VesselAPI tier.

---

## 9. Error handling — surface these as state, not exceptions

| Condition | UI |
|---|---|
| 404 from VesselAPI | Marker fades to stale; tooltip "no recent AIS" |
| 429 rate limit | Banner "VesselAPI throttled — retrying in N s" |
| Network offline | Banner "Offline — showing last known positions" |
| WebSocket disconnect | Auto-reconnect silently after 3 s |
| MMSI unknown for a fleet member | List under "Not yet tracked" in details sheet |

---

## 10. Anti-patterns

- ❌ Storing the API key in `Info.plist` or `.bundleResource`.
- ❌ Matching incoming AIS records by `ShipName` when an MMSI is known.
- ❌ Polling more frequently than every 60 s — you'll hit the free tier
  cap by mid-morning.
- ❌ Showing a "0 kt" arrow for stationary vessels — null out the arrow
  when `speedKnots == 0`.
- ❌ Leaving the WebSocket open while backgrounded — iOS will kill the
  socket and rack up reconnection storms.
- ❌ Using `Date()` for `lastUpdateUTC` if the server timestamp is
  available — that's where staleness rules break down.

---

## 11. Test scenarios

| Scenario | Expected |
|---|---|
| Countess (316004455) — REST query | 200 with `vesselPosition`, or 404 (dockside) |
| Three sister vessels (316004454/455/456) | Often co-located at the same dock; resolving one usually resolves all three |
| Salish Sea Freedom (316042213) in season | Live fix, often Steveston / Plumper Sound |
| Strider (no MMSI) | Filtered out of the map; appears in "no tracking" list |
| Backgrounding the app | Polling pauses, WebSocket closes |
| Foregrounding after 10 min | Single immediate poll, then 60 s cadence |

---

## 12. Required keys

You need **one** of:
- **VesselAPI key** (sign up at https://vesselapi.com — free tier ~250
  requests/day). Pass as `x-api-key`.
- **AISStream key** (sign up at https://aisstream.io — free, no rate
  limit in the bbox). Pass in the subscribe payload.

Two-API plan: use VesselAPI as default, swap to AISStream when the user
opens the "Live tracking" screen.

Store them in **Keychain** using `Security.framework` or the
`SecKey`/`KeychainWrapper` patterns — never `UserDefaults`.
