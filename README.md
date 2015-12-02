# ActorGPU
## What is it?
ActorGPU is an actor-model form on concurrency for the GPU. It uses actor models as a way to simulate concurrency between highly-similar actors. This approach to concurrency allows us to more easily model agent-based modelling on the GPU

##How do I use it?

Easy:

(Code samples coming soon)


## How is this code structured
###Actors
You should be able to subclass actors. Actors are an interface within our system, so when subclassing, you MUST implement a "react" method.
Actors should have a message box within them, or some sense of getting valuable data from others.
This can also be in a public method/private data source, as implemented in some forms of Scala code.

###SchellingActor
SchellingActor is an example of a sub-classing of an Actor.
SchellingActor has two private variables: a type and a number of adjacent actors to it.
SchellingActor has a threshold before it changes places. If there are three or more different around it, then it must change places to a random location on the board.

###MapActor
MapActor is a subclass of Actor.
MapActor keeps track of locations occupied by Actors, and free spaces. MapActor manages positioning of Actors.
MapActor has a public methods freeSpaces() and occupiedSpaces(). These should run kernels to generate an array containing listings of free and occupied spaces, respectively. These are used to determine where to move SchellingActors.
MapActor should deal with receiving signals from SchellingActors telling MapActor they're uncomfortable living in the current situation. When this is received, an atomicCAS should be used to swap positions of SchellingActors around a board.

###!!! ActorSystem
ActorSystem is a generic form of the list of Actors living in a system.
ActorSystem should allow us to be able to find Actors within our system and store them in a sane way
ActorSystem lets us run each Actor's send/receive functions while syncing threads in between.

###Main
Main lets us:

Come up with an ActorSystem

Add MapActors and SchellingActors to our ActorSystem

Run the simulation of having a MapActor keep track of each SchellingActor, and parallelizing the sort algorithm of these Actors.
