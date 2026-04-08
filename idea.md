# AirportOpsEnv — The Full Idea

## What It Is

AirportOpsEnv is an OpenEnv-compliant environment where an AI agent plays the role of an **Airport Ground Operations Controller**. The agent's job is to manage the flow of aircraft on the ground — deciding which planes land on which runways, which gates they park at, and in what order — while simultaneously responding to real-time crisis events like hijackings, medical emergencies, and runway fires.

The environment simulates a task that real humans do every single day at every major airport in the world. That's what makes it score high on real-world utility.

---

## The Core Loop

Every "step" in the environment, the agent receives a snapshot of the airport's current state — which flights are waiting, what resources are free, what time it is, whether there's a crisis active — and must respond with exactly one action: assign a runway, assign a gate, hold a flight, divert one, or scramble an emergency unit. The environment then updates its state, scores the action, and returns a reward. This continues until all flights are handled or the episode times out.

---

## The Airport Resources

The agent manages three types of physical resources. **Runways** — there are three, some dedicated to landing, some to takeoff, some dual-use, and one may be under maintenance in harder scenarios. **Gates** — there are eight, split into types: passenger terminal gates, cargo bays, medical gates, and an isolation bay used exclusively for security threats. The agent must match the right flight to the right gate type. **Ground units** — ambulances, fire trucks, and security teams, each available in limited numbers and dispatched as part of crisis response.

---

## The Priority Hierarchy

Not all flights are equal. The environment has a fixed priority order that the agent must learn and respect. Army and defense flights sit at the top and can preempt anyone. Medevac and medical flights come second. Government and VIP flights are third. Regular commercial passenger flights are fourth. Cargo is last and cannot preempt anyone.

This hierarchy has one important override: if any flight declares a fuel emergency with less than 10 minutes of fuel remaining, it immediately jumps to the top of the queue above even army flights. This is based on real aviation protocol and is the single most important rule the agent must internalize. The grader specifically tests whether the agent handles this correctly.

Time of day and day of week also affect the environment. Rush hour on a Monday morning means longer taxi times and tighter runway availability. A public holiday like Diwali means worse ETAs across the board, a backlog of cargo flights held by the night curfew, and reduced staff availability. These contextual factors are encoded in a pre-computed ETA lookup table, and the agent's gate and runway assignments are scored partly on whether it picked the fastest available option given those conditions.

---

## The Crisis System

What makes this environment genuinely hard — and genuinely novel — is the crisis layer on top of normal operations. Crises activate at specific steps within a scenario and require the agent to switch from optimization mode into protocol-following mode. There are five crisis types.

**Medical emergency** — a flight declares a medical emergency mid-approach. The agent must immediately clear the nearest runway regardless of who's queued ahead of it, assign a medical-type gate (not a cargo bay or standard terminal), and dispatch an ambulance unit. If it fails to do any of these within one step, the protocol score drops significantly.

**Bomb threat** — a flight reports a suspicious package. The agent must route it to the isolation bay, not any passenger gate. It must halt nearby ground movement and notify the correct authorities in the right order: security first, then fire, then police. The airport does not shut down — other runways must stay operational. Sending a bomb threat flight to a passenger terminal results in an immediate hard penalty of zero reward.

**Hijacking** — the flight squawks 7500, the international hijack code. The agent must assign a remote stand far from the terminal and other aircraft. Critically, the action must include a flag called use_secure_channel set to true, because real hijack protocol requires all communication to happen outside public channels. The grader explicitly checks this flag. The agent must also scramble security. Normal operations continue for all other flights — a hijacking does not close the airport.

**Runway fire** — smoke is reported on a landing aircraft or a ground vehicle is on an active runway. The agent must issue a go-around to every aircraft on final approach within one step, dispatch the fire brigade to the correct runway, close that runway in the state so it cannot be used again until cleared, and reroute all pending landings to alternate runways.

**Fuel emergency / Mayday** — a flight reports it has less than 10 minutes of fuel. This overrides every other priority in the system. The agent must assign the closest available runway immediately, even if that means instructing a taxiing aircraft to vacate. Making a Mayday flight wait even one step is a hard penalty.

---

## The Three Tasks

**Task 1 is easy.** Five flights request landing at the same time. One of them is a medevac declaring an emergency mid-approach. The agent must queue the other four in correct priority order and simultaneously clear a runway for the medevac. Resources are abundant, timing is off-peak, and there's only one crisis to handle. This tests basic priority understanding and one straightforward protocol response.

**Task 2 is medium.** Eight flights inbound during Monday morning rush hour. One runway is under maintenance. Three gates are already occupied. A commercial flight has less than 10 minutes of fuel, which means it outranks the army charter also inbound. A cargo plane has a bomb threat. The agent must resolve the conflict between the army flight and the fuel emergency correctly (fuel wins), handle the bomb threat with isolation bay routing and the right authority sequence, and manage the remaining flights without using the closed runway. Resource scarcity and competing priorities are the core challenge.

**Task 3 is hard.** Fifteen simultaneous flight requests on Diwali morning at 6am, just after the night curfew lifted. There is a cargo backlog. One runway is under maintenance. Three gates are occupied by delayed overnight flights. And two crises are active simultaneously — a hijacking and a runway fire happening at the same time. The agent must execute the hijack protocol (remote stand, secure channel, security scramble) and the fire protocol (go-around, fire brigade, runway closure, rerouting) completely independently of each other, without letting one interfere with the other, while still processing all 15 flights as efficiently as possible. This is the scenario that will separate good models from great ones, because most agents will either focus entirely on the crises and stall all normal ops, or handle normal ops and botch one of the protocols.

---

## How the Grader Scores

Every action gets a reward between 0.0 and 1.0 made up of five components. Priority order score checks whether the agent respected the hierarchy for the action it took. Resource match score checks whether the right gate or runway type was assigned to the right kind of flight. ETA optimality score checks whether the chosen assignment was efficient given the time of day and holiday context. Crisis protocol score checks whether the agent followed the correct response sequence for any active crisis. And a penalty component deducts points for hard violations, with some violations like sending a hijacked plane to a passenger terminal setting the total reward to zero regardless of everything else.
