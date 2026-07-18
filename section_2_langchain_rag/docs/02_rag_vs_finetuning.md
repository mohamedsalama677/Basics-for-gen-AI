# Retrieval-Augmented Generation and How It Compares to Fine-Tuning

## The problem RAG was invented to solve

A large language model is only as knowledgeable as the text it was trained on, and once training is finished the model's world view is frozen. If a new product launches next week, the model will not know about it. If a company's internal policy document changes tomorrow, the model will still repeat the old version. Worse, when asked about something outside its knowledge, an LLM does not usually admit ignorance — it invents a plausible answer, a behaviour known as hallucination. Retrieval-Augmented Generation, almost always shortened to RAG, is the pattern that fixes both of these problems by giving the model a way to look things up at answer time instead of relying only on what is baked into its weights.

The idea is straightforward. Before the model generates its answer, a separate retrieval system searches a collection of documents for passages that seem relevant to the user's question. Those passages are then inserted into the prompt, together with an instruction that tells the model to use them as its source of truth. The model still generates the final text, but it is now grounded in evidence that was fresh at the moment of the query. If the evidence changes, so does the answer, and no retraining is needed.

## The RAG pipeline in plain terms

Every RAG system has the same shape. First, the source documents are chunked into small passages, because a full document is too large to feed into the model as context and too coarse to search accurately. Chunking is more subtle than it sounds. Chunks that are too small lose meaning, chunks that are too large blur the retrieval signal, and chunks that ignore natural boundaries like paragraphs or headings tend to cut ideas in half. Overlap between neighbouring chunks helps preserve context that would otherwise be lost at the split.

Second, each chunk is passed through an embedding model, which converts the text into a dense vector that captures its meaning. Similar passages produce similar vectors, and unrelated passages produce vectors that point in different directions. Those vectors are stored in a vector database — FAISS, Chroma, Qdrant, or a hosted equivalent — along with a pointer back to the original chunk.

Third, when a user asks a question, the same embedding model turns the question into a vector, and the vector store returns the chunks whose vectors are closest to it. This is called semantic search, because it retrieves by meaning rather than by exact keyword match. The retrieved chunks are then packed into the prompt in a specific order and passed, together with the question and a system message, to the LLM. The model reads the passages and writes an answer that cites them.

The final step, which is easy to skip and expensive to skip, is handling the case where retrieval fails. If the vector store does not contain anything relevant, a well-designed pipeline refuses to answer instead of letting the model guess. This is the single most effective guard against RAG-flavoured hallucination.

## What fine-tuning does, and where it belongs

Fine-tuning is a different intervention entirely. Where RAG changes what the model sees at inference time, fine-tuning changes the model itself. There are several flavours. Full fine-tuning updates every weight in the network and produces the most thorough change, but it requires enormous compute and easily damages the model's general abilities. Supervised fine-tuning trains the model on curated input-output pairs, teaching it to imitate a specific style, format, or reasoning pattern. Instruction tuning is a form of supervised fine-tuning aimed at making a base model good at following instructions. Parameter-efficient methods like LoRA and other adapters add small trainable matrices on top of a frozen model, giving most of the benefit of fine-tuning at a fraction of the cost and without overwriting the original weights.

Fine-tuning is powerful, but it is a poor fit for facts that change over time. If a fine-tuned model has memorised last quarter's pricing and the price changes, the only way to update it is to gather new training data and run the training job again. That is slow, expensive, and it does not scale to knowledge bases that change every day.

## When to prefer RAG, when to prefer fine-tuning

The choice is not aesthetic; it is driven by what you are trying to change about the model's behaviour. RAG is the right tool when the answer depends on information that is frequently updated, tenant-specific, or must be cited. It shines for customer-support bots that need to quote a policy document, internal search assistants that answer questions about a company wiki, and legal or medical assistants that need every claim traceable to a source. RAG also allows access controls: because retrieval is a separate step, you can restrict which chunks a user is allowed to see before the model ever gets involved.

Fine-tuning is the right tool when the answer depends on how the model behaves rather than what it knows. If a model needs to adopt a specific tone, follow a strict output format, understand a specialised vocabulary that never appears in general web text, or perform a narrow task with lower latency than a long RAG prompt would allow, fine-tuning is often the cleaner solution. It bakes the desired behaviour into the weights so no retrieval step is needed at inference time.

## Combining the two

In practice, mature production systems use both. A model is fine-tuned to speak in the right voice, follow the right output schema, and handle domain-specific vocabulary, and then RAG is layered on top to supply the up-to-date facts. The two techniques are complementary rather than competitive. The rule of thumb is simple: fine-tune for behaviour, retrieve for knowledge.
