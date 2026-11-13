// Apple Vision text recognition for image files: prints "path<TAB>text" per image.
// Usage: swift vision_ocr.swift img1.png img2.png …   (or compile with swiftc for speed)
import Foundation
import Vision
import AppKit

for path in CommandLine.arguments.dropFirst() {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("\(path)\t"); continue
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    let lines = (req.results ?? [])
        .sorted { $0.boundingBox.minX < $1.boundingBox.minX }
        .compactMap { $0.topCandidates(1).first?.string }
    print("\(path)\t\(lines.joined(separator: " | "))")
}
