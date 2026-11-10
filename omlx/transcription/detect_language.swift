import Foundation
import NaturalLanguage

// Local, offline language verification. No microphone or network access.
let input = FileHandle.standardInput.readDataToEndOfFile()
let recognizer = NLLanguageRecognizer()
recognizer.processString(String(data: input, encoding: .utf8) ?? "")
let hypotheses = recognizer.languageHypotheses(withMaximum: 3)
let values = Dictionary(uniqueKeysWithValues: hypotheses.map { ($0.key.rawValue, $0.value) })
FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: values))
