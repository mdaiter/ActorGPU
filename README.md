# ActorGPU
## What is it?
ActorGPU is an actor-model form on concurrency for the GPU. It uses actor models as a way to simulate concurrency between highly-similar actors. This approach to concurrency allows us to more easily model agent-based modelling on the GPU

##How do I use it?

Easy:

(Code samples coming soon)


## How is this code structured
###Actors
You should be able to subclass actors. Actors are an interface within our system, so when subclassing, you MUST implement a "react" method

