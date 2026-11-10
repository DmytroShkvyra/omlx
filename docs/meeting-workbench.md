# Meeting Workbench fork

This fork of [jundot/omlx](https://github.com/jundot/omlx) starts from tag
`v0.7.3` (`9de79bc39a01c1f3fec1e849903aaf0dd5478d7a`). Development branch:
`codex/nemotron-meeting-workbench`. Apache-2.0 and upstream attribution remain.

## Features

- Nemotron whole-recording speaker attribution, including microphone previews
  and a final pass that drains buffered audio when recording stops.
- Separate text translation, Ukrainian by default, with a model selector and
  user-supplied terminology/context. Speaker IDs remain owned by the server.
  Live translation is scheduled after 0.6 seconds of silence following speech,
  or after 8 seconds of continuous speech. Scheduling uses the audio timeline;
  inference already in progress finishes before the newest accumulated prefix
  is processed. Silent input alone does not repeatedly invoke translation.
  Stop always drains the complete received buffer, including a short final tail.
- When translation is unchecked, neither the text translator nor a Nemotron
  translation task runs. Both ordinary ASR and speaker-attributed ASR explicitly
  preserve the original spoken language, including Ukrainian and language switches.
- Optional video/window context, visible speaker names, deduplicated name
  suggestions and manual label corrections.
- File-time frame sampling: uniform pairs, codec keyframes, or meaningful
  scene/region/motion changes. Changes are heuristic, not semantic guarantees.
- Literal URL/URI/ID extraction, including case and punctuation. Partially
  readable values stay marked partial. References are scoped to speech-time
  intervals; unrelated future scenes do not enter the translation context.
- Native CTC timestamps when supported by the checkpoint/runtime; otherwise
  explicitly approximate ASR-window bounds. The tested MLX checkpoint lacks
  the native CTC head.
- Safari microphone activation, silent audio output, a level meter, one Stop
  control, PCM flush acknowledgements and incomplete-result handling.

The UI lives directly in `apps/omlx-web/omlx_web/templates/chat.html`.
Backend code lives in `omlx/transcription/`; it uses the normal audio routers
and their existing authentication. There is no startup template patcher or
separate prototype launcher. ASR, translation and vision share one model pool.
Without visual input, the audio workflow stays unchanged.

## Install and launch from this fork

On an Apple Silicon Mac with macOS 15+, Python 3.11–3.13 and FFmpeg:

```bash
git clone --branch codex/nemotron-meeting-workbench https://github.com/DmytroShkvyra/omlx.git
cd omlx
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[audio]'
brew install ffmpeg
omlx serve --model-dir /path/to/models --base-path "$PWD/.state/server" --port 18082
```

The model directory should contain Nemotron and the selected translator/VLM
(symlinks to existing checkpoints are sufficient). The server model settings
also support multiple model directories. The translation default is `mimo_9b`;
the vision default is `Qwen3.8-27B-MLX-8bit`. Select installed alternatives in
the UI as needed. Microphone/screen APIs need localhost or trusted HTTPS.
Normal oMLX authentication applies; a new installation may require signing in.

Native custom kernels are optional for this installation method; model
families needing them may run slower. With full Xcode and the Metal toolchain,
`make kernels` builds them. `make app` builds a native application bundle;
this fork has not yet published a custom `.app`/DMG release. Upstream DMGs and
Homebrew commands in the original README install upstream, not this branch.

### Build a native `.app` and DMG without full Xcode

The fork also has a SwiftPM manifest and a local packaging script. They compile
the native Swift shell from this branch and reuse an installed official 0.7.3
app for the compatible Python/MLX layers, prebuilt Metal kernels, icon and
localizations. No model weights, chats or runtime settings enter the bundle.

```bash
python3 apps/omlx-mac/Scripts/build_meeting_app.py \
  --donor /path/to/oMLX-0.7.3.app \
  --sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
```

Use the 26.5 SDK with the current Command Line Tools: the newer 27.2 SwiftUI
SDK requires a State macro plugin shipped only with full Xcode. Xcode-only
`#Preview` declarations are excluded from SwiftPM builds. This does not remove
the actual views. Native XCTest suites still require full Xcode (Command Line
Tools do not include the XCTest framework). With Xcode available, configuration
and updater regressions can be run with `swift test --package-path
apps/omlx-mac --sdk <sdk-path> -c release --filter
'AppConfigTests|ReleasesCheckerTests'`. Local CLI-only validation instead checks
the bundle signature, embedded Python imports and an actual app-managed server.

Outputs are `dist/meeting-workbench/oMLX Meeting Workbench.app` and
`dist/meeting-workbench/oMLX-Meeting-Workbench-0.7.3-arm64.dmg`. The local build
is ad-hoc signed and signature-verified, not Apple-notarized. Its runtime OS
requirement follows the donor app; FFmpeg remains an external dependency.

The packaged app has a separate bundle ID, URL scheme, preferences directory
and default data root (`~/.omlx-meeting`). It defaults to port 18081, does not
replace the original `~/.omlx/bin/omlx` CLI, and checks releases only in this
fork. A first Finder launch opens its own setup wizard. To use an existing
configuration or a temporary verification configuration, launch its native
executable with `OMLX_BASE_PATH` pointing at that configuration directory.

By default, translation/vision requests go to this server's own loopback port
with its configured API key. `TRANSLATION_BASE_URL` and
`TRANSLATION_API_KEY` optionally select another compatible server; the local
API key is never forwarded to an explicitly configured external endpoint.
Uploads, microphone spools and the language detector are stored under
`<base-path>/transcription/`, not inside the installed Python package. The
language detector is compiled on demand with Apple's Command Line Tools.

## Validation

```bash
python -B -m unittest discover -s tests/transcription
node tests/transcription/test_visual_ui.cjs
node tests/transcription/test_visual_change_live.cjs
node tests/transcription/test_translation_settings.cjs
node tests/transcription/test_mic_lifecycle.cjs
node tests/transcription/test_mic_stop.cjs
```

The portable regressions exercise whole-recording IDs, Stop buffer draining,
time-scoped context, native/fallback timing, URL/ID preservation, unique names,
translation failure/retry handling and frame-change detection. OpenCV tests
require the audio extra's image dependencies. A real short-audio request on
the fork successfully returned speaker-attributed source text and Ukrainian
translation using Nemotron and `mimo_9b` in the same server. Physical microphone
and screen capture are not claimed by these simulated lifecycle tests.

Migration verification passed 219 selected Python tests (72 meeting tests plus
the upstream chat-ASR, STT and realtime suites; one upstream test excluded by
the existing pytest selection), and all five JavaScript commands above.
Real short-audio checks also passed plain ASR and same-server VLM discovery.

## Updating

Keep `upstream` pointing to `jundot/omlx` and bring upstream updates into this
branch deliberately, running the transcription and upstream audio/UI tests
before changing a working installation. The existing installed oMLX and the
prototype can coexist during migration; switching ports or retiring them is
a separate deployment step. Private runtime settings, model weights, audio,
chat history and verification artifacts are excluded from Git.
