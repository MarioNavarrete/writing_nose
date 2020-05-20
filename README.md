# writing_nose

Using a well known NN trainned on EMNIST dataset + OpenCV, we are able to use our nose to write in front of your laptop!
To try it, first, you need to install all the requirements packages, and then run
> python main.py

It will appear 2 screens, one is a "Mirror" and the other one act like a "Whiteboard". The whiteboard will capture every (x,y) where your nose is, and will join those points with lines.
If you want to erase the drawing, you need to close your eyes as much it takes 4 frames and if you want to save the draw, you need to close your eyes 8 frames: these are parameters, depending on the speed of your camera you can tune it as you wish. The idea of those ones is to be able to blink with normally without delete the progress.

Is just an idea, still a WIP.
