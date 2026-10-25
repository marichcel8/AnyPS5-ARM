import CryptoKit
import Foundation

/// The native engine: aps5-translator turns a title's executable into arm64 code and
/// tools/translator/package_native_app.py makes the bundle, without Rosetta and without the
/// relinker toolkit. It lives in a checkout of the engine repository (branch native-arm64)
/// with its build folders; native-engine.json in Application Support or ANYPS5_NATIVE_ROOT
/// says where.
public struct NativeEngine: Sendable {
    public struct Config: Codable, Equatable, Sendable {
        public var root: String?
        public var translator: String?
        public var libs: String?
        public var vulkan: String?
        public var python: String?
        /// Folders searched for apps the engine already packaged.
        public var importFolders: [String]?

        public init(root: String? = nil, translator: String? = nil, libs: String? = nil, vulkan: String? = nil, python: String? = nil, importFolders: [String]? = nil) {
            self.root = root
            self.translator = translator
            self.libs = libs
            self.vulkan = vulkan
            self.python = python
            self.importFolders = importFolders
        }
    }

    public let root: URL
    public let translator: URL
    public let libs: URL
    public let vulkan: URL
    public let python: URL
    public let version: String

    public static var configFile: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/AnyPS5/native-engine.json")
    }

    public static func loadConfig(from file: URL = NativeEngine.configFile) -> Config {
        guard let data = try? Data(contentsOf: file), let config = try? JSONDecoder().decode(Config.self, from: data) else { return Config() }
        return config
    }

    public var script: URL { root.appendingPathComponent("tools/translator/package_native_app.py") }

    public init(config: Config, environment: [String: String] = ProcessInfo.processInfo.environment) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        func url(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath) }
        let root = url(environment["ANYPS5_NATIVE_ROOT"] ?? config.root ?? home.appendingPathComponent("Developer/AnyPS5-ARM").path)
        self.root = root
        translator = url(environment["ANYPS5_NATIVE_TRANSLATOR"] ?? config.translator ?? root.appendingPathComponent("build-translator").path)
        libs = url(environment["ANYPS5_NATIVE_LIBS"] ?? config.libs ?? root.appendingPathComponent("build-arm64/core/libs/libs").path)
        vulkan = url(environment["ANYPS5_NATIVE_VULKAN"] ?? config.vulkan ?? root.deletingLastPathComponent().appendingPathComponent("AnyPS5-games/vulkan-brew").path)
        python = url(environment["ANYPS5_NATIVE_PYTHON"] ?? config.python ?? "/usr/bin/python3")
        version = NativeEngine.fingerprint(translator: translator, libs: libs)
    }

    /// The engine when its parts are in place, else nil.
    public static func locate(environment: [String: String] = ProcessInfo.processInfo.environment) -> NativeEngine? {
        let engine = NativeEngine(config: loadConfig(), environment: environment)
        return engine.problem == nil ? engine : nil
    }

    /// What is missing, nil when the engine can unlock games.
    public var problem: String? {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: script.path) { return "no tools/translator/package_native_app.py in \(root.path)" }
        if !fileManager.isExecutableFile(atPath: translator.appendingPathComponent("aps5-translator").path) { return "no aps5-translator in \(translator.path)" }
        if !fileManager.fileExists(atPath: translator.appendingPathComponent("libaps5-translator-runtime.a").path) { return "no libaps5-translator-runtime.a in \(translator.path)" }
        let libraries = (try? fileManager.contentsOfDirectory(atPath: libs.path)) ?? []
        if !libraries.contains(where: { $0.hasSuffix(".prx") }) { return "no arm64 libs/*.prx in \(libs.path)" }
        if !fileManager.fileExists(atPath: vulkan.appendingPathComponent("lib/libvulkan.1.dylib").path) { return "no lib/libvulkan.1.dylib in \(vulkan.path)" }
        if !fileManager.isExecutableFile(atPath: python.path) { return "no python3 at \(python.path)" }
        return nil
    }

    /// Changes with the translator, its runtime and the libraries: a game packaged by another
    /// build can be unlocked again.
    static func fingerprint(translator: URL, libs: URL) -> String {
        var hasher = SHA256()
        for file in ["aps5-translator", "libaps5-translator-runtime.a"] {
            try? FileDigest.update(&hasher, with: translator.appendingPathComponent(file))
        }
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: libs.path)) ?? []).filter { $0.hasSuffix(".prx") }.sorted()
        for name in names {
            let values = try? libs.appendingPathComponent(name).resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            hasher.update(data: Data("\(name):\(values?.fileSize ?? 0):\(Int(values?.contentModificationDate?.timeIntervalSince1970 ?? 0))\n".utf8))
        }
        return "native-" + String(FileDigest.hex(hasher.finalize()).prefix(12))
    }

    /// How many lifting and compiling jobs the translator may run at once: each takes about
    /// 1.5 GiB on a large title, and the system and the launcher keep 4 GiB.
    public static func translatorJobs(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory, cores: Int = ProcessInfo.processInfo.activeProcessorCount) -> Int {
        let gibibyte: UInt64 = 1 << 30
        let spare = physicalMemory > 4 * gibibyte ? physicalMemory - 4 * gibibyte : 0
        return max(1, min(cores, Int(spare / (3 * gibibyte / 2))))
    }

    /// What the packaging script runs with: the job count, and Homebrew's LLVM and ld on the path.
    public func environment(base: [String: String] = ProcessInfo.processInfo.environment) -> [String: String] {
        var environment: [String: String] = [:]
        if base["APS5_TRANSLATOR_JOBS"] == nil { environment["APS5_TRANSLATOR_JOBS"] = String(NativeEngine.translatorJobs()) }
        let path = base["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        environment["PATH"] = "/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/local/bin:" + path
        return environment
    }

    /// The arguments package_native_app.py takes for one title.
    public func arguments(elf: URL, game: URL, bundle: URL) -> [String] {
        [script.path, "--eboot", elf.path, "--game", game.path, "--libs", libs.path, "--vulkan", vulkan.path, "--translator", translator.path, bundle.path]
    }

    /// Folders with apps the engine packaged before the launcher unlocked anything.
    public static func importFolders(config: Config = NativeEngine.loadConfig(), games root: URL?) -> [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        var folders = (config.importFolders ?? []).map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        folders.append(home.appendingPathComponent("Developer/AnyPS5-games"))
        if let root { folders.append(root) }
        var seen = Set<String>()
        return folders.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    /// The packaged native apps in `folders` (top level and one folder down), by the title they
    /// hold; the newest wins where a title has several.
    public static func packagedApps(in folders: [URL]) -> [String: URL] {
        let fileManager = FileManager.default
        var found: [String: (URL, Date)] = [:]
        func consider(_ app: URL) {
            guard let titleID = packagedTitle(of: app) else { return }
            let date = (try? app.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            if let known = found[titleID], known.1 >= date { return }
            found[titleID] = (app, date)
        }
        for folder in folders {
            guard let entries = try? fileManager.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { continue }
            for entry in entries {
                if entry.pathExtension == "app" { consider(entry); continue }
                guard (try? entry.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true, entry.lastPathComponent != GameLibrary.storeName else { continue }
                for inner in (try? fileManager.contentsOfDirectory(at: entry, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? [] where inner.pathExtension == "app" { consider(inner) }
            }
        }
        return found.mapValues(\.0)
    }

    /// The title ID of a bundle whose Contents/MacOS/eboot is an arm64 Mach-O and that holds
    /// its game's param.json; nil for anything else.
    static func packagedTitle(of app: URL) -> String? {
        let eboot = app.appendingPathComponent("Contents/MacOS/eboot")
        guard isArm64MachO(eboot),
              let data = try? Data(contentsOf: app.appendingPathComponent("Contents/Resources/game/app0/sce_sys/param.json")),
              let parameters = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let titleID = parameters["titleId"] as? String else { return nil }
        return titleID
    }

    static func isArm64MachO(_ file: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: file) else { return false }
        defer { try? handle.close() }
        let header = handle.readData(ofLength: 8)
        return header == Data([0xCF, 0xFA, 0xED, 0xFE, 0x0C, 0x00, 0x00, 0x01])
    }
}
