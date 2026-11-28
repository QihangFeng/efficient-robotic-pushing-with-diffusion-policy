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

Here’s what I changed. The brown box is our adaptive execution controller. When a plan reaches movement eight, it makes one decision based on the results of camera check and path check. If both checks pass, then it will keep movements nine through twelve from the same prediction. The same PD controller executes them one by one. Otherwise, the it will just replan like the baseline.

Addtionally,  in the originall baseline, the PD controller does not check the camera feedback during executing the prediction of the plan. It just executes the fixed eight movements anyway and then the workflow will ask for a new prediction. While, our adaptive controller will check the camera feedback after each movement to determine whether an early interruption should be triggerd.

## Slide 8 · Camera and path checks

~1min 44s

So, exactly how do the camera and path checks work?

Well, the camera check asks how the push is going. As shown in the table, it checks six signals and all six must pass. 
The signal `Q` represents the current T-block location quality, it tells us whether we can locate the block reliably. 
Based on the signal `Q` result for each movement, the signal `h` counts how many consecutive valid movements we’ve had so far.
So, Together, Q and h help our adaptive controller decide whether it can trust the current coverage estimate.

The signal c measures how much of the target area is covered by the T-block. We set the threshold at 75%. When the coverage reach or exceed this threshold, our adaptive controller will just stop following the current plan and then replan, because, you konw, when the block is near the target position, we typically need more frequent adjustments and timely feedback.

As for the remianing three signals, `W`, `D`, and `R`, I designed them to help tell whether coverage is decreasing or just changing a little as the block moves. W counts drops larger than 1.5 percentage points over the last five movements. We want zero. D tells us how far coverage has fallen from its best reading in the last eight. The limit is 1.5 percentage points. R measures the net change over the last three movements. We allow a drop of at most one percentage point.

Additionally, the path check asks whether the next four movements connect smoothly to the last four. So, it will inspect movements five to twelve, each step must be at least two units of step length and each turn angle should be less than 60 degrees.

## Slide 9 · Recovery fine-tuning

Our second approach is recovery fine-tuning, which updates the policy’s final convolution block, as shown in the orange nodes.

So, we start from a saved state in an failed attempt, then let the original baseline try several continuations and keep the ones that complete the push. Those successful retries become our recovery examples. We mix them with the existing expert demonstrations for fine-tuning.

We train for 600 updates. Only the final convolution block receives updates. The rest of the model stays frozen.

## Slide 10 · Ablation experiment

The ablation experiment shows what each component  contributes. All four configurations use the same 40 starting scenes. Each scene is run twice with with a different random seed each time, so there are 80 attempts for each configuration. We consider a push successful when the coverage exceeds 95 percent within 300 movements.

Let's look at the first row and last row.
So, compared with the original baseline, our combined method raised the success rate from 57.5 to 75 percent, with fewer prediction calls and a shorter runtime. And the average alignment also improved a little.

Now let's look at the last two rows. Fine-tuning alone reaches 70 percent success. Adding adaptive controller saves 18.6 percent of prediction calls, and brings higher success rate. Fewer calls are the clearest benefit of adding the adaptive controller. Recorded runtime also decreased.

## Slide 11 · Standard RL comparison with visual SAC

I also ran a quick comparison with SAC, which is a standard reinforcement learning method. After 100,000 training interactions, it completed zero of the 40 attempts. The PushT-Vision study also reported zero percent success for SAC in its own setup. So my takeaway is that visual PushT was challenging for SAC in both experiments.

## Slide 12 · A saved improvement

Here’s an example where our method helped. Starting from the same scene and initial prediction seed, the original baseline ends at 92 percent coverage after 300 movements. Our method reaches 97 percent in 173 movements, with fewer prediction calls.

## Slide 13 · Limitation of negative flips

While, sometimes the opposite happens. In this example, the original baseline performed very good, but our method failed.
So our methods actually lose some original successful trajectories. This is called negtive flips, which are a well-known issue that can occur when we fine-tune a model.

## Slide 14 · Gained and lost baseline successes

Okay, then, let’s check how often negative flips happen. This table shows how many successes each of the three configurations gained and lost compared with the original baseline.

Let’s look at the net gain column. All three net gains are positive, and the combined method has the largest gain. So, overall, our method gained more successes than it lost and improved the success rate in this experiment.

## Slide 15 · Conclusions

In conclusion, our main contribution is combining adaptive execution with recovery fine-tuningour, and this method achieved a higher success rate with fewer prediction calls.While negative flips are still a limitation. That’s all. Thank you!

## Slide 16 · RL performance and planning costs · Q&A backup

This table comes from a separate PushT-Vision benchmark. After three million environment interactions, five methods recorded zero success in that setup. TD-MPC2 reached 97 percent, and GTP reached 99 percent. Their default planners use learned models to score candidate action sequences and replan after each movement. That gives context for why we studied reusing targets to reduce policy calls. The benchmark uses different movement commands and scoring. Our direct comparison is the visual SAC experiment on slide 11.
