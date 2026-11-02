// Pascaline tokenizer decoding (models/Supra2.5-Pico/tokenization_pascaline.py) from the table written by
// `python -m jxlart.export_web`: drop eos, drop each length token and restore the order of the (reversed)
// digit run after it, concatenate the byte-level BPE byte strings and decode UTF-8 with replacement chars.
import { fetchBytes } from './slm.js';

function b64bytes(s) {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function loadTokenizer(url) {
  const t = JSON.parse(new TextDecoder().decode(await fetchBytes(url)));
  const vocab = t.vocab.map(b64bytes);
  const { eosId, lenBase, lenCount, digitLo, digitHi } = t;
  const utf8 = new TextDecoder('utf-8', { ignoreBOM: true });   // keep U+FEFF like tokenizers' from_utf8_lossy
  const isDigit = (id) => id >= digitLo && id <= digitHi;

  function decode(ids) {
    ids = ids.filter((id) => id !== eosId);
    const plain = [];
    for (let i = 0; i < ids.length;) {
      if (ids[i] >= lenBase && ids[i] < lenBase + lenCount) {
        let j = i + 1;
        while (j < ids.length && isDigit(ids[j])) j++;
        for (let r = j - 1; r > i; r--) plain.push(ids[r]);
        i = j;
      } else plain.push(ids[i++]);
    }
    let n = 0;
    for (const id of plain) n += vocab[id].length;
    const buf = new Uint8Array(n);
    n = 0;
    for (const id of plain) { buf.set(vocab[id], n); n += vocab[id].length; }
    return utf8.decode(buf);
  }

  return { eosId, padId: t.padId, promptIds: t.promptIds, vocabSize: lenBase + lenCount, decode };
}
