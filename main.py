import numpy as np
from ClassFiles.ControlSystemSimulation import ControlSystemSimulation

def main():
    # Specify the order of the system
    system_order = 2
    
    # Sample system matrices (Example for a 2nd order system)
    A = np.array([[0.8, 0.2],
                [-0.1, 0.9]])
    B = np.array([[0.0],
                [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    
    # Set up the simulation
    simulation = ControlSystemSimulation(n=system_order, t_end=50, num_points=1000)
    
    # Build the digital system (from the analog system)
    original_system = simulation.build_digital_system(A, B, C, D)
    
    # Calculate the impulse response of the system
    simulation.impulse_response(original_system)

    # Generate a PWM signal with 1kHz frequency, 50% duty cycle, and 5 seconds duration
    # pwm_signal = simulation.generate_pwm_signal(frequency=1000, duty_cycle=50, duration=5)

    # Generate an exponential swept sine signal from 10Hz to 20kHz over 50 seconds
    swept_sine_signal = simulation.generate_exponential_swept_sine_signal(start_freq=10, end_freq=20e3, duration=50)

    # Simulate the system with the generated swept sine input
    output_signal = simulation.simulate_discrete_state_space(A, B, C, D, swept_sine_signal)

    # Plot the input and output signals
    simulation.plot_input_output(swept_sine_signal, output_signal)
    
    # Ask the user to select a system identification algorithm (SRIM or PEM)
    print("Please choose a system identification algorithm:")
    print("1: SRIM")
    print("2: PEM")
    choice = input("Choose (1 or 2): ")
    
    if choice == '1':
        # Perform system identification using SRIM
        print("Performing system identification using SRIM...")
        identified_system = simulation.identify_system_SRIM(swept_sine_signal, output_signal)
    elif choice == '2':
        # Perform system identification using PEM
        print("Performing system identification using PEM...")
        identified_system = simulation.identify_system_PEM(swept_sine_signal, output_signal)
    else:
        print("Invalid selection. Exiting the program.")
        return
    
    # Plot the step response for both the original and identified systems
    simulation.plot_step_response(original_system, identified_system)
    
    # Plot the eigenvalues for both the original and identified systems
    simulation.plot_eigenvalues(original_system, identified_system)
    
    # Plot the Bode diagrams for both the original and identified systems
    simulation.plot_bode(original_system, identified_system)

if __name__ == "__main__":
    main()
