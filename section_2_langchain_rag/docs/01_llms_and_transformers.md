# Large Language Models and the Transformer Architecture

## What a large language model actually is

A large language model, usually shortened to LLM, is a neural network trained to predict the next token given the tokens that came before it. That single objective — next-token prediction over enormous amounts of text — is deceptively simple, and yet when the network is large enough and the training corpus is broad enough, the resulting model develops surprisingly general capabilities: it can summarise, translate, write code, reason step by step, and hold a conversation. The word "large" in the name is not marketing; it refers to the fact that these models typically contain billions of parameters, and that scale is what turns a text-completion machine into something that behaves like a general-purpose language reasoner.

An LLM does not understand text directly. Before anything can happen, the input string is split into tokens by a tokeniser. A token is usually a sub-word fragment, not a full word, which is why unusual words get chopped into several pieces while common words map to a single token. Each token is then looked up in an embedding table and turned into a dense vector, a list of numbers that carries the token's learned meaning. From that point on, the model works entirely with vectors, not letters.

## The transformer, in prose

Nearly every modern LLM is built on the transformer architecture, introduced in 2017. A transformer is a stack of identical blocks, and each block contains two main ideas: self-attention and a small feed-forward network. Everything else — residual connections, layer normalisation, positional encoding — exists to keep training stable and to help information flow across the stack.

Self-attention is the heart of the architecture. For every token in the input, the model computes three vectors called the query, the key, and the value. The query for one token is compared against the keys of every other token, producing a set of scores that indicate how much attention the current token should pay to each of the others. Those scores are turned into weights, and the values are combined according to those weights. The result is a new representation of the current token that has been contextualised by the rest of the sequence. This mechanism is what lets the model relate a pronoun near the end of a paragraph to the noun near the beginning without any hand-written rule about grammar.

Because a single attention operation captures only one kind of relationship at a time, transformers use multi-head attention. The model runs several attention operations in parallel, each with its own learned projections, and then concatenates and mixes the results. In practice, different heads specialise in different patterns: some track syntax, others track long-range dependencies, others follow entities across a document.

Transformers do not have any built-in notion of word order, so positional encoding is added to the token embeddings before the first block. Early transformers used fixed sinusoidal patterns; modern models often use rotary position embeddings, which encode position by rotating the query and key vectors. Either way, the goal is the same: give the model a way to know that "the dog bit the man" is not the same sentence as "the man bit the dog".

After attention, each block passes its output through a small feed-forward network, applied independently to each position. This is where the model does most of its non-linear thinking. Around both the attention and the feed-forward sub-layers, transformers use residual connections, meaning the input of a sub-layer is added back to its output, and layer normalisation, which rescales activations so training stays stable in deep stacks. Without these two tricks, very deep transformers simply would not train.

## Encoder-only, decoder-only, and encoder-decoder

Not every transformer is used the same way. Encoder-only models, like the original BERT, look at the whole input at once and are good at classification, retrieval, and embedding tasks. Decoder-only models, like the GPT family and Llama, look only at the tokens that came before the current position and are trained to generate text one token at a time. Encoder-decoder models, like the original T5 or many translation systems, use an encoder to read the input and a decoder to write the output, which is a natural fit for sequence-to-sequence tasks such as translation or summarisation. The dominant family for general-purpose LLMs today is decoder-only, because autoregressive generation is what powers chat interfaces.

## Autoregressive generation, sampling, and temperature

When a decoder-only LLM writes a response, it does so one token at a time. After each token is produced, it is appended to the input and the model is asked to predict the next one. The model's output at every step is not a single token but a probability distribution over the entire vocabulary. How that distribution is turned into an actual choice depends on the sampling strategy. Greedy decoding picks the single most likely token, which produces safe but sometimes dull text. Temperature is a knob that flattens or sharpens the distribution: high temperatures make the model more adventurous, low temperatures make it more deterministic. Top-k and top-p (nucleus) sampling restrict the choices to the most probable tokens, which curbs the risk of picking something wildly unlikely.

## Pretraining, post-training, and the context window

An LLM's capabilities come from two stages of training. Pretraining is the expensive stage where the model learns from a very large corpus of text drawn from the web, books, and code. This is where general language ability is built. Post-training is where the model is turned into something useful for people, and it usually includes supervised fine-tuning on curated conversations and reinforcement learning from human feedback, which teaches the model to be helpful, harmless, and honest.

Finally, every LLM has a context window: the maximum number of tokens it can attend to at once. This bounds how much conversation history, retrieved evidence, or source code the model can consider at any moment, and it is one of the most important practical constraints when designing systems on top of an LLM.
