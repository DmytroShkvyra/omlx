// swift-tools-version: 6.0
// SPDX-License-Identifier: Apache-2.0
// Fork modification: build the native shell using Command Line Tools alone.
import PackageDescription

let package = Package(
    name: "oMLXMeetingWorkbench",
    platforms: [.macOS(.v15)],
    products: [.executable(name: "oMLXMeeting", targets: ["oMLX"])],
    dependencies: [
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.0")
    ],
    targets: [
        .executableTarget(
            name: "oMLX",
            dependencies: [.product(name: "MarkdownUI", package: "swift-markdown-ui")],
            path: "Sources"
        ),
        .testTarget(name: "oMLXTests", dependencies: ["oMLX"], path: "Tests/oMLXTests")
    ]
)
