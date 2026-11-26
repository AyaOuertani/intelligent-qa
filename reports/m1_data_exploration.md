# SQuAD 2.0 exploration: findings and decisions


## 1. Splits

| Split | Questions | Articles | Paragraphs | Unanswerable |
| --- | --- | --- | --- | --- |
| train | 117,387 | 398 | 17,134 | 33.1% |
| validation | 12,932 | 44 | 1,895 | 35.7% |
| test (official dev) | 11,873 | 35 | 1,204 | 50.1% |

- 0 answer offsets mismatched; 0 articles shared with the test set.
- **Finding:** test is half unanswerable, train about one third (prior shift).
- **Decision:** always report HasAns / NoAns separately; tune the no-answer threshold on validation by the mean of HasAns F1 and NoAns F1, which does not depend on the class mix (M3, M7).

## 2. Paragraph length

Mean 120 words, median 110, middle 50% between 89 and 143, max 677.

- **Decision (M2):** passages cut from user documents target about 110–150 words, so they look like the paragraphs the reader is trained on.
- **For M7:** most paragraphs plus a question should fit the reader's 384-token window; the long tail (up to 677 words) needs overlapping windows (`doc_stride`). The exact share is measured in tokens in M7.

## 3. Paragraphs per article (document-level retrieval pool)

Mean 43, median 37, range 10–149.

- **Decision (M4):** add a random-retrieval baseline; its expected Recall@1 is the mean of 1/(pool size) per question, so the floor differs per document and must be computed, not assumed.

## 4. Question types

what 59.5%, who 10.2%, how 10.1%, when 6.3%, which 6.3%, where 4.0%, why 1.5%, other 1.4%, whose 0.4%, whom 0.4%.

- Mostly factoid questions (entities, numbers, dates).
- **Limitation:** "why" questions are rare (1.5%), so the reader will be weak on explanation questions that demo users may ask.

## 5. Answer length (answerable questions)

Median 2 words, mean 3.3, 75% of answers are 4 words or fewer, max 43.

- Extractive answers are short spans, which suits EM / token F1.
- **For M7:** a maximum answer length of 30 tokens should cover almost all answers; to be checked in tokens.
- **Note:** the minimum is 0 words because the regex keeps only a–z and 0–9 (answers such as symbols or non-Latin text, and letters like "é", are dropped). **Decision (M2):** the real tokeniser must handle Unicode letters.
- Annotation noise exists, e.g. "How many components are used in an assessment?" → "there are three components of sexual orientation". EM will penalise some correct predictions.

## 6. Question–paragraph word overlap

| | Mean | Median | Middle 50% |
| --- | --- | --- | --- |
| Answerable | 0.67 | 0.67 | 0.50–0.83 |
| Unanswerable | 0.64 | 0.67 | 0.50–0.80 |

- **Finding 1:** about two thirds of a question's content words appear in its own paragraph, and the tallest histogram bar is at 1.0 (all content words copied). This confirms the lexical bias: TF-IDF and BM25 are favoured on SQuAD, and RQ5 (user documents) is needed to check transfer.
- **Finding 2:** unanswerable questions overlap with their paragraph almost exactly as much as answerable ones (they were written to look relevant). Word overlap, and therefore a retrieval score, cannot tell them apart. **Decision:** abstention comes from the reader's no-answer score, not from retrieval scores, as planned.
- The histogram is spiky because overlap is a ratio of small whole numbers (1/2, 2/3, 3/4 ...).
