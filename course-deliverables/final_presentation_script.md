# Final presentation script

**Qihang Feng · ECE 730**

## Slide 1 · Efficient Robotic Pushing with Diffusion Policy

Hi, everyone. I’m Qihang Feng. My project studies how to make better use of a pretrained robot policy.

## Slide 2 · PushT task and feedback loop

The scenario is straightforward. In a simulated environment called PushT, the blue robot pusher moves the gray T-shaped block onto the green target position. The cypher-physic loop has three steps, which are observe, predict, and execute. The model takes camera images and the pusher’s position and predicts where the pusher should move next. After the movements, we get a new camera image and an updated pusher position.

My question is, whether we can complete more pushes while calling the model less often. So, we will focus on the second step, predict, which includes the model and the controller around it.

## Slide 3 · Related work and limitations

There are three related works that motivate our approach.

The Diffusion Policy is where our baseline comes from. It learns from expert demonstrations, so the amount and quality of those demonstrations matter. This motivates our self-imitation fine-tuning method, where we use the model’s own successful retries as extra training examples.

Then, the SGAC generates a new plan at each step of the current plan, and then compares the new plan with current plan to decide which action to use. It makes a new prediction even when it keeps the old plan. This motivated our adaptive controller, which decides dynamically whether to continue current plan or make a new one.

ReGuide also learns from successful recoveries. But it uses a separate model to guide those recoveries, while we simply use the original policy to retry from saved states and keep the continuations that succeed. The difference is, how we recover the failed trajectories, and the Reguide preprint supports the idea of using successful recoveries as extra training data. 

## Slide 4 · What is Diffusion Policy?

We use a  Diffusion Policy model to predict the robot’s movements.
During training, we take clean actions from expert demonstrations and add noise to them. We know what noise we added, so we can teach the model to predict it.
During inference, we start with random action values and repeatedly use the trained model to predict the noise and then remove the predicted noise. Finally, we will get an cleaner movement sequence from the inital noisy start.

## Slide 5 · Why Diffusion Policy?

Then, why diffusion policy? The reason is that, the same situation can have several good solutions. In this example, the demonstrations go around either side of an obstacle. If we use a traditional approach with a loss function like mean squared error, the model can end up averaging the two routes and sending the robot straight through the obstacle.  While Diffusion Policy can learn both options and sample one coherent route. This is the same averaging problem we saw in Exercise 3 of Assignment 2.

## Slide 6 · Original baseline with fixed-eight execution

This is the workflow of the original baseline. One model call generates one prediction, containing 15 usable movements. Each movement is one x-y position. Then it sends the first eight movements to the PD position controller one by one. The PD controller steers the pusher toward each x-y position. When the pusher touches the block, PushT physics will calculate how the block moves. After movement eight, it just discards the unused movements and makes a new plan from fresh observations.

## Slide 7 · Adaptive execution and recovery fine-tuning

Here’s what I changed. The brown box is our adaptive execution controller. When a plan reaches movement eight, it makes one decision. If the camera check and path check both pass, it will keep movements nine through twelve from the same prediction. Then the same PD controller executes them one by one. Otherwise, the it will just replan like the baseline.

Addtionally,  in the originall baseline, the PD controller does not check the camera feedback during executing the prediction of the plan. It just executes the fixed eight movements anyway and then the workflow will ask for a new prediction. Our adaptive controller will check the camera feedback after each movement to determine whether an early interruption should be triggerd.

## Slide 8 · Camera and path checks

The camera check asks how the push is going. We estimate how much of the goal the block covers and look at recent changes. We extend only with valid history, below 75 percent coverage, and within the drop limits on the left. Near the goal, we keep replanning after eight movements.

The path check asks whether the next four targets connect smoothly to the last four. It includes the step from target eight to nine. The checked steps must be at least two workspace units, and turns can be at most 60 degrees. The simulator geometry is used to score the outcome.

## Slide 9 · Recovery fine-tuning

The orange nodes represent our second change. Recovery fine-tuning updates the policy’s final convolution block. I’ll explain how we train it after the controller checks.

I start from a saved state in an attempt that failed. I let the original policy try several continuations and keep the ones that complete the push. Those successful retries become our recovery examples. We mix them with the existing expert demonstrations.

We train for 600 updates, using two expert and two recovery examples per update. Only the orange final convolution block receives updates. The rest of the model stays frozen. We then use the saved checkpoint for evaluation on separate starting scenes.

## Slide 10 · Four matched configurations

This table shows what each change contributed. All four configurations use the same 40 starting scenes. Each scene is run twice with different prediction randomness, so there are 80 attempts per row. A push succeeds when coverage exceeds 95 percent within 300 movements.

Start with the first and last rows. The original policy completed 46 pushes, and the combined method completed 60. That takes observed success from 57.5 to 75 percent, with 22.4 percent fewer prediction calls.

Fine-tuning alone reaches 70 percent success. Now compare the last two rows, which use the same fine-tuned weights. Adding adaptive execution saves 18.6 percent of prediction calls. It gives four more successes, with an uncertainty interval that includes zero. Fewer calls are the clearest benefit of adding the controller. Recorded runtime also fell, with measurements collected across different sessions.

## Slide 11 · Standard RL comparison with visual SAC

I also ran visual SAC, or Soft Actor-Critic, as a standard reinforcement learning baseline. I trained it online for 100,000 interactions with one training seed. The comparison uses the same 40 starting scenes, camera and position inputs, and movement commands.

SAC runs once per scene with a deterministic policy. Diffusion Policy runs twice with different prediction randomness. Our SAC checkpoint completed zero of 40 pushes. This is the result for that checkpoint and training budget. The original Diffusion Policy starts from a model pretrained on expert demonstrations.

## Slide 12 · A saved improvement

Here’s a run where the change helped. Both methods start from the same scene and initial prediction seed. The original ends at 92 percent coverage after 300 movements. The combined method reaches 97 percent in 173 movements, with 20 prediction calls compared with 38 for the original.

## Slide 13 · Limitation of negative flips

Sometimes the opposite happens. Here the original policy succeeds at about 98 percent coverage. The combined method ends at about 82 percent after 300 movements. We call this a negative flip because a push that succeeded under the baseline fails under the new method. The reason for this failure is still unresolved.

## Slide 14 · Gained and lost baseline successes

To see how often this happens, I paired each run with the original policy on the same scene and prediction draw. A gain is a baseline failure that the new method completes. A loss is a push that only the baseline completes. The combined method gains 23 successes and loses nine. That’s a net gain of 14, which takes us from 46 to 60 successes.

## Slide 15 · Conclusions

The combined method raised observed success from 57.5 to 75 percent in PushT. Adding adaptive execution to the fine-tuned policy saved 18.6 percent of prediction calls. We still lost nine baseline successes, so preserving those pushes is the next step. Thank you for listening.

## Slide 16 · RL performance and planning costs · Q&A backup

This table comes from a separate PushT-Vision benchmark. After three million environment interactions, five methods recorded zero success in that setup. TD-MPC2 reached 97 percent, and GTP reached 99 percent. Their default planners use learned models to score candidate action sequences and replan after each movement. That gives context for why we studied reusing targets to reduce policy calls. The benchmark uses different movement commands and scoring. Our direct comparison is the visual SAC experiment on slide 11.