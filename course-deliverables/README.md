# Course deliverables

This folder holds the ECE 730 proposal, mid-project update and future final-report materials.

- `project-proposal.pdf`: published one-page proposal.
- `proposal-presentation.pdf`: published seven-slide proposal presentation, including the closing thank-you slide.
- `proposal_script.pdf`: slide-by-slide speaking script for an approximately five-minute presentation.
- `mid-update-presentation.pdf`: fifteen-slide presentation with the task and research question, Diffusion Policy denoising and motivation, related work, adaptive execution and controller checks, recovery fine-tuning, evaluation setup, four-configuration ablation, visual SAC comparison, saved examples, gained and lost successes, and conclusions.
- `assets/`: conceptual illustrations and the recorded seed-0 PushT animation used in the presentation.

The editable `project-proposal.docx`, `proposal-presentation.pptx`, `proposal_script.docx` and `mid-update-presentation.pptx` sources stay local and are ignored by Git. Both local presentations include speaking notes. The mid-update notes also identify experiment sources and the limits of the comparisons.

The proposal materials describe the research question, proposed method and evaluation plan. They intentionally omit completed experiment results. The research evidence remains in `reports/` and `results/` at the repository root.

The presentation shows the [four-configuration results](../reports/2026-11-03-robotics-four-configuration-ablation.md) and [visual SAC comparison](../reports/2026-11-06-robotics-visual-sac-comparison.md). Its example images reproduce saved confirmation recordings on scenes 201 and 179, draw offset 60000. Every reconstructed camera frame matches its saved RGB hash. The local PowerPoint includes a looping ten-second seed-0 recording; the published PDF shows a static frame.

After the SAC comparison, the gained/lost-success table compares each configuration with the original fixed-eight policy on the eighty matched scene/prediction-seed pairs. The combined method gains 23 successes and loses nine, for a net gain of fourteen. Investigating those nine losses is a proposed next step; any recovery change needs separate evaluation scenes.

The motivation diagram illustrates multiple valid action sequences with a conceptual obstacle-avoidance example. The related-work page summarizes Diffusion Policy, SGAC and ReGuide with their reported limitations. A separate RL page reproduces GTP's published success table and explains default per-movement planning. That literature table uses its authors' training budget, velocity commands and IoU scoring. The project's comparison uses the forty shared PushT starting scenes. Runtime claims against external planners require matched measurements. Source links and interpretation limits appear in the speaker notes.

The implementation reuses Diffusion Policy and the PushT environment. Source references and illustration credits appear in the document or relevant slide notes. The conceptual illustrations use AI-generated artwork and are explanatory diagrams. ChatGPT assisted with drafting and artifact preparation. Qihang should review the materials and complete any instructor-required reflection before submission.
