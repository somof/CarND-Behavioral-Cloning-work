That's cool. What did you do for braking etc? I use a very simple bit of logic:

    if np.abs(steering_angle) > 0.7:
        controller.set_desired(9)
    elif np.abs(steering_angle) > 0.5:
        controller.set_desired(15)
    elif np.abs(steering_angle) > 0.2:
        controller.set_desired(19)
    else:
        controller.set_desired(30)
Where the numbers where just adjusted by hand. I did have to add one more bit of logic:

    if speed > (controller.set_point + 10):
        throttle -= 100
        print("Slamming on brakes!")
This was needed for the bit right near end because we're going downhill and then there's a sudden turn. So our speed is much higher than our desired speed, and the PI controller isn't aggressive enough.

