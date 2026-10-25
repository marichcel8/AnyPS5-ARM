import Foundation
import XCTest
@testable import AnyPS5Core

final class NativeEngineTests: XCTestCase {
    private var folder: URL!

    override func setUpWithError() throws {
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("native-engine-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: folder)
    }

    private func fakeApp(_ name: String, titleID: String, arm64: Bool = true) throws -> URL {
        let app = folder.appendingPathComponent(name + ".app")
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: app.appendingPathComponent("Contents/MacOS"), withIntermediateDirectories: true)
        try fileManager.createDirectory(at: app.appendingPathComponent("Contents/Resources/game/app0/sce_sys"), withIntermediateDirectories: true)
        let header: [UInt8] = arm64 ? [0xCF, 0xFA, 0xED, 0xFE, 0x0C, 0x00, 0x00, 0x01] : [0xCF, 0xFA, 0xED, 0xFE, 0x07, 0x00, 0x00, 0x01]
        try Data(header + [0, 0, 0, 0]).write(to: app.appendingPathComponent("Contents/MacOS/eboot"))
        try Data(#"{"titleId":"\#(titleID)"}"#.utf8).write(to: app.appendingPathComponent("Contents/Resources/game/app0/sce_sys/param.json"))
        return app
    }

    func testJobsFollowMemory() {
        let gibibyte: UInt64 = 1 << 30
        XCTAssertEqual(NativeEngine.translatorJobs(physicalMemory: 8 * gibibyte, cores: 8), 2)
        XCTAssertEqual(NativeEngine.translatorJobs(physicalMemory: 4 * gibibyte, cores: 8), 1)
        XCTAssertEqual(NativeEngine.translatorJobs(physicalMemory: 32 * gibibyte, cores: 8), 8)
        XCTAssertEqual(NativeEngine.translatorJobs(physicalMemory: 32 * gibibyte, cores: 4), 4)
    }

    func testPackagedArm64AppsAreFoundByTitle() throws {
        let doom = try fakeApp("DOOM (metal)", titleID: "PPSA99666")
        _ = try fakeApp("Rosetta", titleID: "PPSA00001", arm64: false)
        let found = NativeEngine.packagedApps(in: [folder])
        XCTAssertEqual(found["PPSA99666"]?.standardizedFileURL, doom.standardizedFileURL)
        XCTAssertNil(found["PPSA00001"])
    }

    func testImportedBundleCountsAsInstalledWithoutAnInstallRecord() throws {
        let app = try fakeApp("DOOM", titleID: "PPSA99666")
        let game = Game(folder: folder.appendingPathComponent("dump"), titleID: "PPSA99666", name: "DOOM", version: "1.0", icon: nil, background: nil, music: nil, executable: .elf)
        Installer.refreshImports(for: [game], root: folder, folders: [folder])
        defer { Installer.refreshImports(for: [], root: folder, folders: []) }
        XCTAssertEqual(Installer.bundle(for: game, root: folder).standardizedFileURL, app.standardizedFileURL)
        XCTAssertEqual(Installer.record(of: app)?.toolkit, Installer.importedToolkit)
        XCTAssertTrue(Installer.isImported(game))
    }

    func testArgumentsNameEveryInput() {
        let engine = NativeEngine(config: .init(root: "/r", translator: "/t", libs: "/l", vulkan: "/v", python: "/p"), environment: [:])
        let arguments = engine.arguments(elf: URL(fileURLWithPath: "/e.elf"), game: URL(fileURLWithPath: "/g"), bundle: URL(fileURLWithPath: "/b.app"))
        XCTAssertEqual(arguments, ["/r/tools/translator/package_native_app.py", "--eboot", "/e.elf", "--game", "/g", "--libs", "/l", "--vulkan", "/v", "--translator", "/t", "/b.app"])
    }
}
