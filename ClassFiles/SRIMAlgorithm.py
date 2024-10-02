import numpy as np

class SRIMAlgorithm:
    def __init__(self, n, m, r, l, Y, U, mode):
        """
        Initialize the class
        n: System order
        m: Number of outputs
        r: Number of inputs
        l: Length of data
        Y: Output data matrix (m x l)
        U: Input data matrix (r x l)
        """
        self.n = n
        self.m = m
        self.r = r
        self.l = l
        self.Y = Y
        self.U = U
        self.p = self.select_p()
        self.mode = mode

    def select_p(self):
        """
        Select the parameter p
        p >= n / m + 1
        """
        p = int(np.ceil(self.n / self.m) + 1 + 10)
        # print(f"Selected p: {p}")
        return p

    def construct_Yp_Up(self):
        """
        Create the output data matrix Yp and input data matrix Up
        """
        N = self.l - self.p + 1
        Yp = np.zeros((self.p * self.m, N))
        Up = np.zeros((self.p * self.r, N))

        for i in range(N):
            Yp[:, i] = self.Y[i:i+self.p].flatten('F')
            Up[:, i] = self.U[i:i+self.p].flatten('F')

        # print(f"Constructed Yp: {Yp}")
        # print(f"Constructed Up: {Up}")
        return Yp, Up

    def calculate_correlation_matrices(self, Yp, Up):
        """
        Calculate correlation matrices Ryy, Ryu, Ruu
        """
        N = Yp.shape[1]
        Ryy = (1 / N) * np.dot(Yp, Yp.T)
        Ryu = (1 / N) * np.dot(Yp, Up.T)
        Ruu = (1 / N) * np.dot(Up, Up.T)

        # print(f"Calculated Ryy: {Ryy.shape}")
        # print(f"Calculated Ryu: {Ryu.shape}")
        # print(f"Calculated Ruu: {Ruu.shape}")
        return Ryy, Ryu, Ruu

    def calculate_Rhh(self, Ryy, Ryu, Ruu):
        """
        Calculate the correlation matrix Rhh
        """
        Ruu_inv = np.linalg.pinv(Ruu)  # Use pseudo-inverse in case Ruu is non-invertible
        Rhh = Ryy - np.dot(np.dot(Ryu, Ruu_inv), Ryu.T)

        # print(f"Calculated Rhh: {Rhh.shape}")
        return Rhh

    def singular_value_decomposition(self, Rhh):
        """
        Perform singular value decomposition and create the observation matrix O_p
        """
        U, S, Vh = np.linalg.svd(Rhh)
        U_n = U[:, :self.n]  # Extract the part corresponding to the n singular values
        U_o = U[:, self.n:]
        O_p = U_n
        
        '''
        print(f"U: {U}")
        print(f"U.shape: {U.shape}")
        print(f"U_n: {U_n}")
        print(f"U_n.shape: {U_n.shape}")
        print(f"U_o: {U_o}")
        print(f"U_o.shape: {U_o.shape}")
        print(f"Singular values: {S}")
        print(f"U_n (Observation matrix O_p): {O_p.shape}")
        '''
        
        return O_p, U_o

    def calculate_A_matrix(self, O_p):
        """
        Calculate the state matrix A
        """
        # print(f"O_p (before A calculation): {O_p.shape}")
        A = np.dot(np.linalg.pinv(O_p[1:(self.p-1)*self.m, :]), O_p[self.m+1:self.p*self.m, :])
        # print(f"Calculated A matrix: {A.shape}")
        return A
    
    def generate_U_on(self, U_n, U_o):
        m = self.m
        p = self.p
        # Get the dimensions of U_o
        pm, n_o = U_o.shape
        # print(f"pm: {pm}, n_o: {n_o}")

        # Get the dimensions of U_n
        pm_n, n = U_n.shape
        # print(f"pm_n: {pm_n}, n: {n}")
        
        # Empty lists to create the submatrices
        U_o_blocks = []
        U_n_blocks = []

        # Generate block matrices from U_o (transpose considered)
        U_o_transposed = U_o.T  # Transpose
        # print("U_o Transposed:")
        # print(U_o_transposed)

        for i in range(p):
            start_col = i * m
            end_col = (i + 1) * m
            U_o_block = U_o_transposed[:, start_col:end_col]
            U_o_blocks.append(U_o_block)
            # print(f"U_o_block {i+1}:")
            # print(U_o_block)

        # Block matrices generated from U_n (using the transposed U_o)
        for i in range(1, p):
            start_col = i * m
            U_n_block = U_o_transposed[:, start_col:pm] @ U_n[0:(p-i)*m, :]
            U_n_blocks.append(U_n_block)
            # print(f"U_n_block {i}:")
            # print(U_n_block)

        # The last block is a zero matrix
        zero_block = np.zeros((n_o, n))
        U_n_blocks.append(zero_block)
        # print(f"Zero block:")
        # print(zero_block)

        # Construct the final U_on matrix
        U_on = []
        for i in range(p):
            combined_block = np.hstack((U_o_blocks[i], U_n_blocks[i]))
            U_on.append(combined_block)
            # print(f"Combined block {i+1}:")
            # print(combined_block)

        # Vertically stack to generate the final matrix
        U_on = np.vstack(U_on)
        # print("Final U_on matrix:")
        # print(U_on)

        return U_on

    def calculate_u_oT(self, U_oR):
        r = self.r
        # Get the dimensions of the matrix
        n_o, pr = U_oR.shape
        p = pr // r  # Calculate the number of blocks

        # Slice the matrix by blocks and stack them vertically
        U_oT = np.vstack([U_oR[:, i*r:(i+1)*r] for i in range(p)])
        
        return U_oT

    def calculate_B_D_matrices(self, O_p, U_o, Ryu, Ruu):
        """
        Calculate the input matrix B and direct transmission matrix D
        """
        # print(f"U_o (before B and D calculation): {U_o.shape}")
        U_oR = np.dot(np.dot(U_o.T, Ryu), np.linalg.pinv(Ruu))

        # print(f"U_oR matrix: {U_oR.shape}")
        # print(f"U_oR content: {U_oR}")

        U_on = self.generate_U_on(O_p, U_o)

        # print(f"U_on matrix: {U_on.shape}")
        # print(f"U_on content: {U_on}")

        U_oT = self.calculate_u_oT(U_oR)

        # print(f"U_oT matrix: {U_oT.shape}")
        # print(f"U_oT content: {U_oT}")

        # Calculate the BD matrix (adjust shapes of O_p and U_oR)
        BD = np.dot(np.linalg.pinv(U_on), U_oT)  # Adjust U_oR shape

        # print(f"BD matrix: {BD.shape}")
        # print(f"BD content: {BD}")

        if self.mode == 0:
            # Adjust the shape of D to (m × r)
            D = BD[:self.m, :].reshape(self.m, self.r)
        else:
            D = np.zeros((self.m, self.r))
        # Adjust the shape of B to (n × r)
        B = BD[self.m:, :].reshape(self.n, self.r)

        # print(f"Calculated D matrix: {D.shape}")
        # print(f"Calculated B matrix: {B.shape}")
        return B, D

    def calculate_C_matrix(self, O_p):
        """
        Calculate the output matrix C
        """
        if self.mode == 0:
            # print(f"O_p (before C calculation): {O_p.shape}")
            # Extract the first m rows of O_p and reshape to (m, n)
            C = O_p[:self.m, :].reshape(self.m, self.n)
            # print(f"Calculated C matrix: {C.shape}")
        else:
            C = np.zeros((self.m, self.n))
            C[0, 0] = 1
        return C

    def run(self):
        """
        Run the entire SRIM algorithm
        """
        Yp, Up = self.construct_Yp_Up()
        Ryy, Ryu, Ruu = self.calculate_correlation_matrices(Yp, Up)
        Rhh = self.calculate_Rhh(Ryy, Ryu, Ruu)
        O_p, U_o = self.singular_value_decomposition(Rhh)
        A = self.calculate_A_matrix(O_p)
        C = self.calculate_C_matrix(O_p)
        B, D = self.calculate_B_D_matrices(O_p, U_o, Ryu, Ruu)

        return A, B, C, D
