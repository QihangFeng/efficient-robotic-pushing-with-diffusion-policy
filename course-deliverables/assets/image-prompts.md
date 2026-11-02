# Illustration sources and prompts

Both PNG files were generated with the built-in image-generation tool for this proposal presentation. They are conceptual illustrations, not measured robot trajectories, model architecture specifications or experimental results. The presentation also uses cropped simulator illustrations from the existing project evidence; their paths and source credits appear in the relevant slide notes.

## Action reuse

Output: `action-reuse.png`

Prompt:

Use case: scientific-educational. Asset: original conceptual illustration for a robotics course proposal presentation. Wide landscape image approximately 16:9, pure white background, restrained navy #142D4E, teal #167C80 and amber #C98D37, crisp academic infographic with flat 2D vector-like clarity rendered as bitmap. Two spacious horizontal action timelines. TOP ROW exact title 'Fixed eight'. Draw EXACTLY EIGHT separate teal circular action markers, numbered 1, 2, 3, 4, 5, 6, 7, 8 in order, then a simple camera icon and a new-prediction symbol. BOTTOM ROW exact title 'Adaptive eight or twelve'. Draw exactly EIGHT teal circular action markers numbered 1 through 8, then a clearly visible decision/checkpoint after marker 8. After that checkpoint draw exactly FOUR amber circular cached-action markers numbered 9, 10, 11, 12. Put exact short text 'Continue or replan' near checkpoint. The twelve numbered markers in lower row must remain in reading order; decision mark must sit between 8 and 9 without replacing a numbered marker. The lower row can include one branch leading to the camera/new-prediction symbol, but continuation remains the straight path. Large clean sans-serif labels, ample whitespace. No results, charts, success percentages, robot arms, language-model text, gradients, shadows, dashboards, decorative cards, watermarks, or extra text. This explains execution length, not generation length. Count the eight and four markers carefully.

## Policy refinement

Output: `policy-refinement.png`

Prompt:

Use case: scientific-educational. Asset: original conceptual diagram for an academic robotics proposal slide, landscape approximately 16:9, pure white background, navy #142D4E, teal #167C80, amber #C98D37. Explain bounded offline refinement of an existing vision-based robot-action model. Left side contains two small 2D training-example groups: a small T-shaped block and circular pusher with a short movement trace in each group. Label upper group exactly 'Expert examples' and lower group exactly 'Recovery examples'. A subtle clean connector leads from both groups toward a large pale-gray frozen neural-network backbone in the center. Label this central network exactly 'Frozen backbone' and show small lock symbol. Attached on its right is a much smaller amber final convolution output block, labeled exactly 'Trainable output block'. Only this last block is amber and trainable; all earlier layers remain gray and frozen. On right show a short sequence of 2D movement coordinates as teal dots, labeled exactly 'Movement predictions'. Flat technical illustration, generous whitespace, large legible sans-serif labels, accurate simple neural diagram. No 3D robot arm, no LLM, no results, no percentages, no measured memory, no charts, no decorative dashboard/UI, no fancy shadows, no extra text. Illustration depicts offline training, not online candidate generation.
