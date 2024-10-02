import numpy as np

# A class for system identification using the Predictive Error Method
class PredictiveErrorMethod:

    def __init__(self, input_data, output_data, system_order):
        """
        Initialization function. Sets the input signal, output signal, and system order.
        """
        self.input_data = input_data
        self.output_data = output_data
        self.system_order = system_order

    def estimate_ab_coefficients(self):
        """
        Estimates the a and b coefficients of the discrete-time transfer function using 
        the least squares method.
        
        Returns:
        --------
        a : np.ndarray
            Coefficient vector for the denominator of the transfer function.
        b : np.ndarray
            Coefficient vector for the numerator of the transfer function.
        """
        N = len(self.output_data)
        n = self.system_order
        m = n - 1  # Typically, m = n - 1 as per the given document

        # Construct Omega matrix for the least squares problem
        Omega = []
        y_vector = []
        
        for k in range(n, N):
            # Ensure we are within the valid index range for both input and output signals
            if k >= n and k >= m:
                # Construct each row of Omega
                y_terms = -self.output_data[k-1:k-n-1:-1] if k-n >= 0 else []
                u_terms = self.input_data[k:k-m-1:-1] if k-m >= 0 else []

                # Check if both y_terms and u_terms are valid and of the correct length
                if len(y_terms) == n and len(u_terms) == m + 1:
                    Omega.append(np.hstack([y_terms, u_terms]))
                    # Output data (y) from k to N
                    y_vector.append(self.output_data[k])

        Omega = np.array(Omega)
        y_vector = np.array(y_vector)

        # Solve the least squares problem to find the coefficients
        theta = np.linalg.inv(Omega.T @ Omega) @ Omega.T @ y_vector
        
        # Split theta into a (denominator) and b (numerator) coefficients
        a = theta[:n]
        b = theta[n:]
        
        return a, b

    def transfer_function_to_state_space(self, a, b):
        """
        Converts transfer function coefficients to state-space representation using 
        the Controllable Canonical Form.

        Parameters:
        -----------
        a : np.ndarray
            Coefficient vector for the denominator of the transfer function.
        b : np.ndarray
            Coefficient vector for the numerator of the transfer function.
        
        Returns:
        --------
        A : np.ndarray
            State matrix.
        B : np.ndarray
            Input matrix.
        C : np.ndarray
            Output matrix.
        D : np.ndarray
            Feedthrough matrix.
        """
        n = len(a)

        # Construct matrix A (State matrix)
        A = np.zeros((n, n))
        A[:-1, 1:] = np.eye(n - 1)
        A[-1, :] = -a[::-1]  # The last row contains the negative a coefficients

        # Construct matrix B (Input matrix)
        B = np.zeros((n, 1))
        B[-1, 0] = 1

        # Construct matrix C (Output matrix)
        C = b[::-1].reshape(1, -1)

        # Construct matrix D (Feedthrough matrix)
        D = np.zeros((1, 1))  # Typically D is zero for this case

        return A, B, C, D

    def identify_state_space(self):
        """
        Combines the estimation of a and b coefficients and the conversion to state-space
        matrices A, B, C, D.
        
        Returns:
        --------
        A : np.ndarray
            State matrix.
        B : np.ndarray
            Input matrix.
        C : np.ndarray
            Output matrix.
        D : np.ndarray
            Feedthrough matrix.
        """
        # Step 1: Estimate the transfer function coefficients a and b
        a, b = self.estimate_ab_coefficients()

        # Step 2: Convert the transfer function to state-space representation
        A, B, C, D = self.transfer_function_to_state_space(a, b)

        return A, B, C, D
