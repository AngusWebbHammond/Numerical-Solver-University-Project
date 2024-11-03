# Numerical Solver
import numpy as np
import matplotlib.pyplot as plt
class Solver_Normal:
    def __init__(self, a, d, k, m, h, points, boundary, critical=False, times=0.2, loops=1000):
        # Initialising the Class with any variables needed throughout
        self.a = a
        self.d = d
        self.k = k
        self.m = m
        self.h = h
        self.points = points
        self.boundary = boundary
        self.critical = critical
        self.delta_r = 0.1
        self.delta_t = self.delta_r ** 2 * times
        self.loops = loops
        self.test = np.indices((1, points))
        self.start_f = np.array(self.test[1][0])
        self.f_updated = np.array(self.test[1][0])
        self.r_list = np.array(self.test[1][0])
        self.t_list = np.array(np.indices((1, self.loops))[1][0])
        self.energy_exact = []
        self.f_exact = []
        self.f_several = []
        self.energy = []
        # Finding the Starting values/Exact values
        self.start()
        self.exact()
        # Calling the algorithm for the solver
        self.loop()
    def start(self):
        # Starting Values
        self.start_f = self.boundary[0] * (1 - self.r_list / (self.points - 1))
        self.f_updated = self.boundary[0] * (1 - self.r_list / (self.points - 1))
        self.r_list = self.r_list * self.delta_r
        self.t_list = self.t_list * self.delta_t
    def derivative(self, function):
        # 1st Derivative Calculation
        function_i_plus1 = np.append(function[1:], self.boundary[0])
        function_i_minus1 = np.append(self.boundary[1], function[0:self.points - 1])
        function_derivative = (function_i_plus1 - function_i_minus1) / (2 * self.delta_r)
        return function_derivative
    def derivative_2(self, function):
        # Second Derivative Calculation
        function_1 = function
        function_i_plus1 = np.append(function[1:], self.boundary[0])
        function_i_minus1 = np.append(self.boundary[1], function[0:self.points - 1])
        function_derivative_2 = (function_i_plus1 + function_i_minus1 - 2 * function_1) / (self.delta_r ** 2)
        return function_derivative_2
    def step(self, function):
        # The Main Calculation for f_t and Energy
        function_1 = function[1:self.points - 1]
        function_unchanged = function
        cos_function = np.cos(function_1)
        sin_function = np.sin(function_1)
        r_values = self.r_list[1:self.points - 1]
        derivative = self.derivative(function_unchanged)[1:self.points - 1]
        derivative_2 = self.derivative_2(function_unchanged)[1:self.points - 1]
        if not self.critical:
            # When MH!=-2K
            f_t = 2 * (self.a * (derivative_2 + derivative / r_values - sin_function * cos_function / (
                    r_values ** 2)) + self.d * sin_function ** 2 / r_values - self.k * sin_function * cos_function - \
                       self.m * self.h * sin_function / 2)
            energy_vals = 2 * np.pi * (
                    self.a * (derivative ** 2 + sin_function ** 2 / (r_values ** 2)) + self.d * (1 - cos_function) * (
                    derivative - sin_function / r_values) + self.k * sin_function ** 2 + self.m * self.h * (
                            1 - cos_function)) * r_values * self.delta_r
        else:
            # Used when MH=-2K
            f_t = 2 * (self.a * (derivative_2 + derivative / r_values - sin_function * cos_function / (
                    r_values ** 2)) + self.d * sin_function ** 2 / r_values - np.abs(self.k) * (
                                   1 - cos_function) * sin_function)
            energy_vals = 2 * np.pi * (
                    self.a * (derivative ** 2 + sin_function ** 2 / (r_values ** 2)) + self.d * (1 - cos_function) * (
                    derivative - sin_function / r_values) + np.abs(self.k) * (
                            1 - cos_function) ** 2) * r_values * self.delta_r
        self.energy.append(sum(energy_vals))
        return function_1 + self.delta_t * f_t
    def loop(self):
        # The Main Loop
        for a in range(self.loops):
            self.f_updated = self.step(self.f_updated)
            self.f_updated = np.append(self.boundary[0], self.f_updated)
            self.f_updated = np.append(self.f_updated, self.boundary[1])
    def exact(self):
        # Exact Solutions when MH=-2K
        f_exact = np.append(self.boundary[0], 2 * np.arctan(self.d / (np.abs(self.k) * self.r_list[1:])))
        self.energy_exact = np.zeros(self.loops) + 8 * np.pi * (self.a - self.d ** 2 / (2 * np.abs(self.k)))
        self.f_exact = f_exact
# Coordinate Transfomation
class Solver_Transformation:
    def __init__(self, a, d, k, m, h, points, boundary, critical=False,loops=1000):
        # Initialising the Class with any variables needed throughout
        self.a = a
        self.d = d
        self.k = k
        self.m = m
        self.h = h
        self.points = points
        self.boundary = boundary
        self.critical = critical
        self.delta_u = 1 / (self.points - 1)
        self.delta_t = self.delta_u ** 2 * 0.3
        self.loops = loops
        self.f_list = []
        self.loop_list = [10000,15000,20000,25000,30000,35000,40000]
        self.test = np.indices((1, points))
        self.start_f = np.array(self.test[1][0])
        self.f_updated = np.array(self.test[1][0])
        self.u_list = np.array(self.test[1][0]) * self.delta_u
        self.r_list = np.zeros(self.points)+np.tan(np.pi * self.u_list / 2)
        self.t_list = np.array(np.indices((1, self.loops))[1][0]) * self.delta_t
        self.energy_exact = []
        self.f_exact = []
        self.energy = []
        # Finding the Starting values/Exact values
        self.start()
        self.exact()
        # Calling the Main Loop
        self.loop()
    def start(self):
        # Starting Values
        self.start_f = self.boundary[0] * (1 - self.u_list / self.delta_u / (self.points - 1))
        self.f_updated = self.boundary[0] * (1 - self.u_list / self.delta_u / (self.points - 1))
    def derivative(self, function):
        # 1st Derivative Calculation
        function_i_plus1 = np.append(function[1:], self.boundary[0])
        function_i_minus1 = np.append(self.boundary[1], function[0: - 1])
        function_derivative = (self.points - 1) * (function_i_plus1 - function_i_minus1) / 2
        return function_derivative
    def derivative_2(self, function):
        # 2nd Derivative Calculation
        function_1 = function
        function_i_plus1 = np.append(function[1:], self.boundary[0])
        function_i_minus1 = np.append(self.boundary[1], function[0: - 1])
        function_derivative_2 = (function_i_plus1 + function_i_minus1 - 2 * function_1) * (self.points - 1) ** 2
        return function_derivative_2
    def step(self, function):
        # The Main Calculation for f_t and Energy
        function_1 = function[1:- 1]
        function_unchanged = function
        cos_function = np.cos(function_1)
        sin_function = np.sin(function_1)
        u_vals = self.u_list[1:-1]
        cos_2 = np.cos(np.pi / 2 * u_vals)
        sin_2 = np.sin(np.pi / 2 * u_vals)
        du_dr = 2 / np.pi * cos_2 ** 2
        derivative = self.derivative(function_unchanged)[1:- 1]
        derivative_2 = self.derivative_2(function_unchanged)[1:- 1]
        jacobian = np.pi / 2 * np.tan(np.pi / 2 * u_vals) * 1 / (cos_2 ** 2)
        tan_vals = np.tan(np.pi / 2 * u_vals)
        if not self.critical:
            # When MH!=-2K
            energy_vals = 2 * np.pi * (
                    2 * self.a * du_dr ** 2 * (derivative ** 2 + sin_function ** 2 / tan_vals ** 2) + self.d * (
                    1 - cos_function) * (du_dr * derivative - sin_function / tan_vals) + np.abs(self.k) * (
                            1 - cos_function) ** 2) * jacobian * self.delta_u
        else:
            # Used When MH=-2K
            f_t = 2 * self.a * (du_dr**2 * derivative_2 + 2 / np.pi * cos_2 ** 3 * derivative * (
                        1 / sin_2 - 2 * sin_2) - sin_function * cos_function / (
                                            sin_2 ** 2/cos_2**2)) + 2 * self.d * sin_function ** 2 / (sin_2/cos_2) - 2 * np.abs(
                self.k) * sin_function * (1 - cos_function)
            energy_vals = 2 * np.pi * (
                    self.a * ((du_dr * derivative) ** 2 + sin_function ** 2 / tan_vals ** 2) + self.d * (
                        1 - cos_function) * (
                            du_dr * derivative - sin_function / tan_vals) + np.abs(self.k) * (
                            1 - cos_function) ** 2) * jacobian * self.delta_u
        self.energy.append(sum(energy_vals))
        return function_1 + self.delta_t * f_t
    def loop(self):
        # The Main Loop
        for a in range(self.loops):
            self.f_updated = self.step(self.f_updated)
            self.f_updated = np.append(self.boundary[0], self.f_updated)
            self.f_updated = np.append(self.f_updated, self.start_f[-1])
    def exact(self):
        # Finding Exact Solution for MH=-2K
        f_exact = np.append(self.boundary[0], 2 * np.arctan(self.d / (np.abs(self.k) * self.r_list[1:])))
        self.energy_exact = np.zeros(self.loops) + 8 * np.pi * (self.a - self.d ** 2 / (2 * np.abs(self.k)))
        self.f_exact = f_exact
# Domain Wall Solver
class Solver_DW:
    def __init__(self, a, k, points, boundary, real=False, c=1/1000, b=1, alpha=1, mu=1, current=1, current_allow=False,
                 loops=1000,
                 factor=0.05):
        # Initialising the Class with any variables needed throughout
        self.a = a
        self.k = k
        self.points = points
        self.boundary = boundary
        self.real = real
        self.alpha = alpha
        self.mu = mu
        self.c = c
        self.b = b
        self.current = current
        self.current_allow = current_allow
        self.delta_x = 0.05
        self.delta_t = self.delta_x ** 2 * factor
        self.loops = loops
        self.test = np.indices((1, points))
        self.start_theta = np.array(self.test[1][0] - (self.points - 1) / 2)
        self.start_phi = np.array(self.test[1][0] - (self.points - 1) / 2)
        self.phi_updated = np.array(self.test[1][0] - (self.points - 1) / 2)
        self.theta_updated = np.array(self.test[1][0] - (self.points - 1) / 2)
        self.x_list = np.array(self.test[1][0] - (self.points - 1) / 2)
        self.t_list = np.array(np.indices((1, self.loops))[1][0] - (self.points - 1) / 2)
        self.exact_theta = []
        self.exact_energy = 0
        self.energy = []
        self.theta_list = []
        # Starting/Exact Value Calling
        self.start()
        self.exact()
        # Calling Main Loop
        self.loop()
    def start(self):
        # Starting Values
        self.start_phi = np.zeros(self.points)
        self.start_theta = self.boundary[0] / 2 * (1 - 2 * self.x_list / (self.points - 1))
        self.phi_updated = np.zeros(self.points)
        self.theta_updated = self.boundary[0] / 2 * (1 - 2 * self.x_list / (self.points - 1))
        self.x_list = self.x_list * self.delta_x
        self.t_list = self.t_list * self.delta_t
    def derivative(self, function):
        # 1st Derivative
        function_i_plus1 = np.append(function[1:], self.boundary[0])
        function_i_minus1 = np.append(self.boundary[1], function[0:self.points - 1])
        function_derivative = (function_i_plus1 - function_i_minus1) / (2 * self.delta_x)
        return function_derivative
    def derivative_2(self, function):
        # 2nd Derivative
        function_1 = function
        function_i_plus1 = np.append(function[1:], self.boundary[0])
        function_i_minus1 = np.append(self.boundary[1], function[0:self.points - 1])
        function_derivative_2 = (function_i_plus1 + function_i_minus1 - 2 * function_1) / (self.delta_x ** 2)
        return function_derivative_2
    def step(self, functions):
        # Main Calculation of theta_t and phi_t
        function_1 = functions[0][1:self.points - 1]
        function_2 = functions[1][1:self.points - 1]
        function_1_unchanged = functions[0]
        function_2_unchanged = functions[1]
        cos_function_1 = np.cos(function_1)
        sin_function_1 = np.sin(function_1)
        derivative_1 = self.derivative(function_1_unchanged)[1:self.points - 1]
        derivative_2 = self.derivative(function_2_unchanged)[1:self.points - 1]
        second_derivative_1 = self.derivative_2(function_1_unchanged)[1:self.points - 1]
        second_derivative_2 = self.derivative_2(function_2_unchanged)[1:self.points - 1]
        if self.real:
            # When using the Landau-Lifschitz Gilbert equation
            dot_theta = 2 / self.mu * (self.a * (
                        second_derivative_1 - sin_function_1 * cos_function_1 * derivative_2 ** 2) - self.k * sin_function_1 * cos_function_1)
            dot_phi = 2 / self.mu * (self.a * (
                        sin_function_1 * second_derivative_2 + 2 * cos_function_1 * derivative_1 * derivative_2))
            theta_t = 1 / (1 + self.alpha ** 2) * dot_phi + self.alpha / (1 + self.alpha ** 2) * dot_theta
            phi_t = 1 / sin_function_1 * (
                        -1 / (1 + self.alpha ** 2) * dot_theta + self.alpha / (1 + self.alpha ** 2) * dot_phi)
        else:
            # When using Gradient Flow
            theta_t = 2 * self.a * (
                    second_derivative_1 - cos_function_1 * sin_function_1 * derivative_2 ** 2) - 2 * self.k * sin_function_1 * cos_function_1
            phi_t = 2 * self.a * (
                    sin_function_1 ** 2 * second_derivative_2 + 2 * sin_function_1 * cos_function_1 * derivative_1 * derivative_2)
        if self.current_allow:
            # When considering a current density
            theta_t += self.current * (self.c/self.mu * derivative_2 * sin_function_1 + self.b/self.mu * derivative_1)
            phi_t += self.current * (-self.c/self.mu * derivative_1 + self.b/self.mu * derivative_2 * sin_function_1)
        energy_vals = (self.a * (
                derivative_1 ** 2 + sin_function_1 ** 2 * derivative_2 ** 2) + self.k * sin_function_1 ** 2) * self.delta_x
        self.energy.append(sum(energy_vals))
        function_1 += self.delta_t * theta_t
        function_2 += self.delta_t * phi_t
        return function_1, function_2
    def loop(self):
        # The Main Loop
        for a in range(self.loops):
            functions = self.step([self.theta_updated, self.phi_updated])
            np.append(self.boundary[0], np.append(functions[0], self.boundary[1]))
            np.append(self.boundary[0], np.append(functions[1], self.boundary[1]))
    def exact(self):
        # Exact Values when MH=-2K
        exact_theta = 2 * np.arctan(np.exp(-np.sqrt(self.k / self.a) * self.x_list))
        self.exact_theta = exact_theta
        self.exact_energy = 4 * np.sqrt(self.a * self.k)