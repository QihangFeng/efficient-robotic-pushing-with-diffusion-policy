# Course deliverables

This folder holds the ECE 730 proposal, mid-project update and future final-report materials.

- `project-proposal.pdf`: published one-page proposal.
- `proposal-presentation.pdf`: published seven-slide proposal presentation, including the closing thank-you slide.
- `proposal_script.pdf`: slide-by-slide speaking script for an approximately five-minute presentation.
- `mid-update-presentation.pdf`: ten-slide progress update with the feedback loop, adaptive execution, recovery fine-tuning, matched results, visual SAC comparison, saved gain and regression, and questions for the instructor.
- `assets/`: conceptual illustrations embedded in the presentation.

The editable `project-proposal.docx`, `proposal-presentation.pptx`, `proposal_script.docx` and `mid-update-presentation.pptx` sources stay local and are ignored by Git. Both local presentations include speaking notes. The mid-update notes also identify experiment sources and the limits of the comparisons.

The proposal materials describe the research question, proposed method and evaluation plan. They intentionally omit completed experiment results. The research evidence remains in `reports/` and `results/` at the repository root.

The mid-update presents the [four-configuration results](../reports/2026-11-03-robotics-four-configuration-ablation.md) and [visual SAC comparison](../reports/2026-11-06-robotics-visual-sac-comparison.md). Its example images reproduce saved confirmation recordings on scenes 201 and 179, draw offset 60000. Every reconstructed camera frame matches its saved RGB hash. The ten slides support about ten minutes of speaking within the twenty-minute progress discussion.

The implementation reuses Diffusion Policy and the PushT environment. Source references and illustration credits appear in the document or relevant slide notes. The two conceptual illustrations were generated with AI assistance; they are explanatory diagrams, not experiment records. ChatGPT assisted with drafting and artifact preparation. Qihang should review the materials and complete any instructor-required reflection before submission.
