# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the colongrade documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **image** | One colonoscopy frame (or one synthetic drawing) with one grade. | picture, frame (alone), sample |
| **grade** | The Mayo endoscopic subscore of an image: 0, 1, 2 or 3. | class, label (alone), severity score |
| **active disease** | Grade 2 or 3. Remission is grade 0 or 1. | inflamed, positive |
| **manifest** | `manifest.csv`: one row per image with path, grade, split, patient and fold. | index, image list |
| **split** | `trainval` or `test`, from the dataset folders. | partition, set (alone) |
| **fold** | One of the grouped cross-validation parts of `trainval`. | split (for a fold), chunk |
| **patient** | The patient id from a folder name, if the dataset has one. | case, subject |
| **contrast mode** | `none`, `equalize` or `clahe`, applied to the luminance only. | filter, enhancement |
| **grader** | A model that gives four grade probabilities. | classifier (alone), network |
| **feature grader** | `features-multinomial` or `features-ordinal` on colour and texture features. | baseline model, classic model |
| **CNN grader** | `cnn-small`, `cnn-resnet18` or `cnn-resnet50` with a CORAL head. | deep model |
| **CORAL head** | One shared score with three ordered thresholds for P(grade > k). | ordinal layer |
| **conditional GAN** | The generator and discriminator that draw images for a given grade. | DCGAN, GAN (alone) |
| **generated image** | An image from the conditional GAN. It is used only in training folds. | fake image, synthetic image |
| **synthetic image** | A drawing from `colongrade synth`, for the demo and the tests. | generated image, fake data |
| **run** | One experiment folder with config, results, model and model card. | job, trial |
| **QWK** | Quadratic weighted kappa, the main metric for ordered grades. | kappa (alone), agreement |
| **model card** | `model_card.md`: intended use, data, results and limits of a run. | report, datasheet |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **grade** | Give an image four grade probabilities and the most likely grade. |
| **list** | Write the manifest of all images. |
| **validate** | Score a model on its held-out fold. |
| **evaluate** | Score the final model on the test split one time. |
| **generate** | Draw images for given grades with the conditional GAN. |
| **equalize** | Spread the luminance histogram of an image. |
