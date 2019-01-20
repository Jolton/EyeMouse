# EyeMouse

## Inspiration
You that annoying moment when you have 3 monitors and you begin typing and realize the words are not showing up? You have this sudden panic of 'Oh no, where am I actually typing!'

Our Goal was to solve this problem

## What it should do
It will track your eye movement and bring to the foreground any window that you are looking at.

## How we built it
We used Python and OpenCV to do computer vision on the webcam feed of a computer.

## Challenges we ran into
Not knowing much computer vision, we ran into a lot of confusion about how to actually recognize the eye. Second is the methods we used proved to be inaccurate at best and controlling the location of the eye on the screen is still a challenge we have to solve. There are many variables we can fine tune to produce a more accurate detection, however being unfamiliar with the underlying computer vision algorithms we were picking values at random and hoping the result would be better.

## Accomplishments that we're proud of
That we attempted something challenging and interesting! We also learned how to pick out a face and eyes in an image using openCV which was super cool!

## What we learned
Webcams have low resolution and the simple idea of tracking where your eyes are looking is a huge undertaking and not easily done by a novice. Python is also annoying when variables randomly change type and you have no idea why!

## What's next for EyeMouse
Ideally we would expand it to use actual eye trackers. From our minimal research, these seem to use a combination of a much higher resolution image and infrared illumination to more accurately detect the pupil. There are also more sophisticated libraries available for such products and we predict we could be much more accurate.


## References
https://picoledelimao.github.io/blog/2017/01/28/eyeball-tracking-for-mouse-control-in-opencv/
https://docs.opencv.org/
