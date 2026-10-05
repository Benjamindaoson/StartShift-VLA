# Research reflection: why this project stops here

StartShift-VLA made initialization robustness a concrete reliability concern. A policy that succeeded in 86 of 120 selected canonical-reset episodes succeeded in only 36 of 800 selected RobotInit episodes. That difference is useful evidence to preserve, even though the evaluation does not isolate reset state from task composition.

The project is stopping because its next defensible claim would require a substantially stronger evidence base. Continuing to develop adapters on top of the current comparison would risk optimizing a proposed solution before establishing which failures it should solve. The RISE variants remain unvalidated prototypes; their incompleteness is part of the record, not something to hide behind a paper-style presentation.

Four gaps matter most. First, paired evaluation should hold task, objects, scene and policy randomness constant while varying a verified robot reset intervention. Second, multiple models are needed to distinguish a checkpoint-specific limitation from a broader phenomenon. Third, a failure taxonomy needs traceable videos and state/action records so that reaching, perception, contact, control and timeout failures are not conflated. Fourth, deployment-oriented experiments must evaluate practical recovery and reposition strategies under realistic cost and operating constraints.

None of these requirements was satisfied by a successful shell command or by a larger unpaired evaluation. The training path also stopped at missing episode-to-variant provenance. That was a reason to withhold a method claim, not a negative result for the architecture.

The decision is therefore to close StartShift-VLA as a research archive. It preserves the baseline measurements, code, design alternatives, failed setup attempts and explicit missing evidence. It is not an active plan to finish the RISE training matrix.

Future effort is redirected toward **World Model Reliability / Reliable Physical Intelligence**. The transferable question is how an embodied system can recognize when its predictions or learned behavior stop being reliable under changing physical conditions, and how that uncertainty should inform action, recovery and deployment decisions. Reset variation is one instance of that broader problem. This direction names an area for future research; it does not claim a completed world model or reliable robot system.
