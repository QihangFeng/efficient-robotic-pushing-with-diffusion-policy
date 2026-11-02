# Illustration sources and prompts

The original PNG illustrations were generated with the built-in image-generation tool. The current action diagram was edited with that tool; the current fine-tuning diagram combines a generated neural-network illustration with user-approved precise SVG scene panels. They are conceptual illustrations, not measured robot trajectories, model architecture specifications or experimental results. The presentation also uses cropped simulator illustrations from the existing project evidence; their paths and source credits appear in the relevant slide notes. Revision details and exact prompts appear below.

## Original action-reuse illustration

Output: `action-reuse.png`

Prompt:

Use case: scientific-educational. Asset: original conceptual illustration for a robotics course proposal presentation. Wide landscape image approximately 16:9, pure white background, restrained navy #142D4E, teal #167C80 and amber #C98D37, crisp academic infographic with flat 2D vector-like clarity rendered as bitmap. Two spacious horizontal action timelines. TOP ROW exact title 'Fixed eight'. Draw EXACTLY EIGHT separate teal circular action markers, numbered 1, 2, 3, 4, 5, 6, 7, 8 in order, then a simple camera icon and a new-prediction symbol. BOTTOM ROW exact title 'Adaptive eight or twelve'. Draw exactly EIGHT teal circular action markers numbered 1 through 8, then a clearly visible decision/checkpoint after marker 8. After that checkpoint draw exactly FOUR amber circular cached-action markers numbered 9, 10, 11, 12. Put exact short text 'Continue or replan' near checkpoint. The twelve numbered markers in lower row must remain in reading order; decision mark must sit between 8 and 9 without replacing a numbered marker. The lower row can include one branch leading to the camera/new-prediction symbol, but continuation remains the straight path. Large clean sans-serif labels, ample whitespace. No results, charts, success percentages, robot arms, language-model text, gradients, shadows, dashboards, decorative cards, watermarks, or extra text. This explains execution length, not generation length. Count the eight and four markers carefully.

## Original policy-refinement illustration

Output: `policy-refinement.png`

Prompt:

Use case: scientific-educational. Asset: original conceptual diagram for an academic robotics proposal slide, landscape approximately 16:9, pure white background, navy #142D4E, teal #167C80, amber #C98D37. Explain bounded offline refinement of an existing vision-based robot-action model. Left side contains two small 2D training-example groups: a small T-shaped block and circular pusher with a short movement trace in each group. Label upper group exactly 'Expert examples' and lower group exactly 'Recovery examples'. A subtle clean connector leads from both groups toward a large pale-gray frozen neural-network backbone in the center. Label this central network exactly 'Frozen backbone' and show small lock symbol. Attached on its right is a much smaller amber final convolution output block, labeled exactly 'Trainable output block'. Only this last block is amber and trainable; all earlier layers remain gray and frozen. On right show a short sequence of 2D movement coordinates as teal dots, labeled exactly 'Movement predictions'. Flat technical illustration, generous whitespace, large legible sans-serif labels, accurate simple neural diagram. No 3D robot arm, no LLM, no results, no percentages, no measured memory, no charts, no decorative dashboard/UI, no fancy shadows, no extra text. Illustration depicts offline training, not online candidate generation.


## Revised presentation illustrations (2 November 2026)

The current `action-reuse.png` is a built-in image-generation edit of the original illustration. The two report/chart icons were replaced with matching connected-node symbols labeled “Action model.” Camera symbols, action numbering and decision branches retain their original meaning.

The current `policy-refinement.png` combines an AI-generated neural-network illustration with precise SVG scene panels. Qihang explicitly approved vector redrawing after reviewing inconsistent T proportions in the generated versions. The editable source is `policy-refinement.svg`; its neural-network background is embedded, so it can be opened independently.

All eight scene panels reuse one 104-by-104 T-shaped path, without scaling, for both the block and target. Each target keeps the same relative pose. Angled block poses, curved approaches and pusher positions preserve the preferred earlier example. Pusher circles remain outside the object. Final aligned frames indicate completion and do not depict further actions. These frames are schematic examples, not recorded simulator trajectories or exact physical predictions.

The recovery diagram branches from one saved intermediate state into an unsuccessful original continuation and a successful retry using the same original model. Only the successful continuation connects to training, alongside expert examples. The diagram does not imply that all retries succeed or that failure trajectories supply supervised action targets.

### Action-model icon edit — exact prompt

Use case: precise-object-edit, scientific-educational. Asset: the existing academic robotics execution illustration. Image 1 is the edit target.
Change ONLY the two document/bar-chart icons, one at the far right of the top row and one after the lower camera icon. Replace each with the SAME simple navy neural-network symbol (three columns of small connected nodes), clearly labeled "Action model". This represents using Diffusion Policy to generate a new movement prediction, not an evaluation report. Keep the network symbols compact and aligned to the existing incoming arrows, with labels beneath.
Strict invariants: preserve the wide canvas, pure white background, navy/teal/amber palette, all existing text, camera symbols, arrows, and exact circle positions and numbers. Top row must remain EXACTLY eight teal numbered circles 1 through 8 under "Fixed eight". Bottom row must remain eight teal circles 1 through 8, then the "Continue or replan" diamond, then four amber circles 9, 10, 11, 12 under "Adaptive eight or twelve". Preserve the curved branch from the diamond to camera and then action model. Do not add a chatbot, brain character, report icon, extra circles, decorative framing, or red annotation boxes. Preserve all margins and avoid clipping labels.

### Training-example edit — exact prompt

Use case: precise-object-edit, scientific-educational. Asset: update the existing academic robot policy fine-tuning illustration. Image 1 is the EDIT TARGET. Image 2 is the user's hand-drawn STRUCTURE REFERENCE, not an edit target.
Primary request: correct ONLY the story of the training inputs while retaining the existing white/navy/teal/gray/amber style and the frozen-backbone, trainable-output-block and movement-predictions explanation on the right. A wide landscape canvas about 2.4:1 will fit the slide. Give the input examples about 55-60% of the width so the recovery branch is readable. The right pipeline may compact to make room, but keep the same gray 3D layer-stack appearance, navy lock, amber final output block and teal output dots. Use large legible sans-serif labels and sparse arrows.
EXPERT INPUT: top-left group titled exactly "Expert examples". Three equal sequential scene frames. In EVERY frame draw the SAME pale green dashed T-shaped TARGET outline at the same relative position and orientation. The movable solid GRAY T-shaped block changes its position/angle across the frames: initially offset and rotated from target, then closer after a push, then aligned over the target with the green outline still visible. A TEAL circular pusher changes position and visibly pushes the gray block toward the target. Keep its circle outside the block, in contact when pushing. These are three moments of one pushing process, not three unrelated demonstrations.
RECOVERY INPUT: bottom-left group titled exactly "Recovery examples". Show ONE shared scene on the left, labeled "Saved intermediate state", with fixed target, misaligned gray block and teal pusher. From THAT EXACT SHARED STATE visibly fork into two short chronological continuations. Upper path labeled "Original continuation" has one further pushing frame and a final MISALIGNED block frame marked with a small muted-red X and "Failure". Lower path labeled "Resampled continuation" has one further pushing frame and a final ALIGNED target/block frame marked with a green check and "Success". Both branches use the original model. The lower path restarts from the shared intermediate state, NOT from the failed endpoint. Keep the identical target outline visible in all recovery frames; move the block and pusher appropriately. Label only the selected lower successful continuation "Selected trajectory".
TRAINING WIRES: Expert examples and ONLY the successful lower recovery continuation feed the neural training pipeline. The failed upper branch must end at its Failure marker and must NOT have an outgoing arrow to training. Connect from the successful branch rather than a generic arrow from the whole recovery panel. Exact right-side labels remain "Frozen backbone", "Trainable output block", "Movement predictions". Only the final output block is amber/trainable; earlier layers remain gray/frozen with a lock. No success rates, charts, measurements, LLM graphics, red annotation rectangles, or claims of measured results. These are conceptual example frames, not simulator screenshots. Do not omit, move, or rotate the target between frames.

### Caption revision — exact prompt

Use case: precise-object-edit. Image 1 is the edit target: the already corrected policy-refinement illustration.
Make ONLY a readability adjustment to the small recovery captions. Replace "Saved intermediate state" with "Saved state", "Original continuation" with "Original", and "Resampled continuation" with "Retry". Enlarge these three captions plus "Failure", "Success", and "Selected trajectory" so their font size is approximately the same as the "Movement predictions" label, with no overlap. Short labels should be clear when the whole diagram is displayed on one presentation slide.
Strictly preserve the existing wide canvas, all scene frames, all fixed green dashed target outlines, gray moving T blocks, teal pushers, the exact saved-state fork and arrow routes, failure X, success check, expert group, training selection bracket, successful-only training connector, frozen backbone with lock, amber output block, output movement dots, white background and navy/teal/gray/amber colors. Keep all other text unchanged. Do not add frames, change any scene geometry, or connect the failed branch to training.

### Precise scene correction

User-approved vector editing preserved the generated right-hand neural-network pipeline and redrew only the example panels and their input connectors. The shared T template, poses and contact geometry are encoded directly in the SVG. No image-generation prompt was used for this final geometry correction. The diagram is a conceptual explanation of expert/recovery data and limited fine-tuning, not an architecture specification or an experimental result.

The recovery panel was subsequently extended downward, with more space between the scene panels, selection bracket and outer border. This SVG spacing adjustment preserves all scene geometry and the neural-network illustration.

## Presentation style reference

The green-and-gold palette, header bands and University of Alberta wordmark are adapted from the user-provided `uofa_presentation_template.pptx`. The cover follows Qihang's selected split green-and-gold reference, with the template's patterned A artwork and unmodified white wordmark. The closing slide retains the green wordmark. The method illustrations remain unchanged; no new illustrations were generated for this styling revision.

## Fixed-eight discarded predictions (2 November 2026)

Output: `action-reuse-with-discard.png`. Generated with the built-in image tool as a separate version of the original image. The fixed-eight row now shows cached actions 9–15 as grey unexecuted predictions discarded after action 8. The accepted result was visually checked for the complete 1–8 sequences in both rows. This version replaces the illustration on presentation slide 2, with the matching PDF refreshed; the other six slides retain their existing appearance.

### Initial edit prompt

Use case: precise-object-edit / educational infographic.
Edit target: the supplied two-row robotic action-execution diagram. Regenerate a polished updated version, retaining its white background, bold navy sans-serif headings, thin navy arrows, teal numbered executed actions, amber optional actions, camera icons, and connected-node action-model icons.

Primary change: explicitly show the fixed-eight baseline DISCARDING the remaining predictions after executing action 8.
Top section heading exactly "Fixed eight". Preserve the main executed sequence 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8, then camera icon → neural-network icon labeled "Action model". Add a clearly separate downward fork immediately after action 8. This fork must be a pale-grey DASHED connector to pale-grey unexecuted predicted-action circles labeled "9", "10", "…", "15". Add a visible muted-red X / cancellation mark over the grey predicted sequence, and label it exactly "Discard remaining predictions". This grey fork represents actions that were predicted but NEVER EXECUTED. The live replanning path remains navy. Do not color the discarded actions teal or amber. Do not suggest the robot executes them.

Bottom section heading exactly "Adaptive eight or twelve". Preserve the executed teal sequence 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Then the diamond marked "?" with heading "Continue or replan", a straight continuation branch to amber circles 9 → 10 → 11 → 12, and a curved lower replanning branch to camera icon → neural-network icon labeled "Action model".

Layout: expand the white canvas vertically as necessary so the new grey branch occupies generous whitespace BETWEEN sections, without crowding or overlapping either section. Keep all numbers, labels, arrows, and icons crisp and legible. Preserve the established professional minimalist slide-illustration appearance. No surrounding slide, no watermark, no extra text, no ornamental graphics.

### Numbering correction prompt

Edit the supplied diagram. Correct ONLY the numbered executed-action sequences: the current image accidentally omits the circle labeled 4 in BOTH rows. Each row MUST contain EXACTLY EIGHT TEAL CIRCLES before the camera or decision diamond, labeled in this exact left-to-right order:
1, 2, 3, 4, 5, 6, 7, 8.
Insert the missing 4 between 3 and 5 in BOTH rows. Redistribute the eight teal circles evenly over the current available horizontal space; shrink their diameter slightly if needed. Draw all eight, do not merge or omit any. Verify visually that each row reads 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.
Keep every other feature and exact text unchanged: fixed-eight live path to camera and Action model; grey dashed discard branch 9,10,…,15 and red X and "Discard remaining predictions"; adaptive diamond "Continue or replan" and amber9,10,11,12 and curved replan-camera-model branch. Preserve white backdrop, navy type/arrows/icons, teal and amber colors, spacious layout, and crisp infographic style. No additional decoration.
