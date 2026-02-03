# Aira.js-Preview

**Aira is a artificial intelligence(GPT algorithm) library that is currently under development.**

By running on the GPU, it delivers performance optimizations for tensor operations and training loops.

## Tests
**- Intel® HD Graphics 630**

 **- 221 lines dataset, 128 hidden, 64 embedSize, 3 epochs, learning rate 0.003**
    ![inteltest](inteltest.png)

## Features

MLP Layer: Multilayer perceptron with forward and backward propagation

MHA (Multi-Head Attention): Transformer-based attention mechanism

LayerNorm: Normalization for training stability

Tensor Operations: GPU-accelerated matrix and vector multiplications

NLP Modules: BPE tokenization with encode/decode support

Training Loop: Training an MLP model on data with loss tracking

Sampling: Top-K and Top-P sampling, repeat penalty, and temperature control

Note: A WebGPU-enabled browser is required to run this project (Chrome Canary or the latest versions of Edge).

## Quick Start
```html
<script type="module" src="src/Aira.js">

(async () => {
  const device = await ensureGPU();
  const embedSize = 128
  const hiddenSize = 512
  const trainedModel = await StartModel("http://127.0.0.1:5500/datasets/en_Aira_big_chat_ds.txt", 500, embedSize, hiddenSize, 5, 0.003);
  const usrprompt = "Hello!";
  const mha = new MHA(device,embedSize,3)
  const layernorm = new LayerNorm(embedSize)
  const response = await generateReply(
    mha,
    trainedModel.mlp, 
    layernorm, 
    trainedModel.nlp, 
    trainedModel.embed, 
    usrprompt
  );
  
  console.log("Aira:", response);
})();
  
</script>
```
