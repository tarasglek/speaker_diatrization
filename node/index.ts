import { parseSync, NodeList } from 'subtitle'
import { promises as fs } from "fs";
// import microsoft-cognitiveservices-speech-sdk
import { SpeechConfig, SpeechSynthesizer, AudioConfig } from "microsoft-cognitiveservices-speech-sdk"

async function main() {
    let input = await fs.readFile(process.argv[2], 'utf-8')
    let srt: NodeList = parseSync(input)
    // print srt to stdout
    for (let i = 0; i < 4 /*srt.length*/; i++) {
        let node = srt[i]
        if (node.type !== 'cue') {
            continue
        }
        let d = node.data
        let text = d.text.
            replace('[Music]', '').
            trim()
        if (text.length === 0) {
            continue
        }
        console.log(JSON.stringify(text))
    }

}

main()