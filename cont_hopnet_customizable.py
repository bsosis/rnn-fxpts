import numpy as np
import sys



def _update_sync(W, gain, a):
        """Update the current activation"""

        return np.tanh(gain*np.dot(W, a))

def _energy_sync(W, gain, cur_act, prev_act):
        """
        Uses eq. 3.4 in Soulie et al. 1989
        But modifies eq. 3.2 to include gain, and adds gain term to first sum in eq. 3.4 (equivalent to just multiplying the weights by the gain before use).
        If prev_act is None, tries to invert W to find the energy.
        """

        if prev_act is None:
            W_inv = np.linalg.inv(W)
            # Test that W_inv can actually successfully invert W
            if not np.allclose(np.eye(W.shape[0]), np.matmul(W, W_inv), rtol=0, atol=1e-7):
                return float('nan') # TODO: pseudoinverse?
            prev_act = np.dot(W_inv, np.arctanh(cur_act)/gain)


        cur_field = gain*np.dot(W,cur_act)
        prev_field = gain*np.dot(W,prev_act)

        # summation = prev_act^T * W * cur_act
        summation = gain*np.matmul(prev_act[None,:], np.matmul(W, cur_act[:,None]))

        # Let c=0; note that log(cosh(0))=0

        # sum_int1 = sum(log(cosh(prev_field)))
        sum_int1 = np.sum(np.log(np.cosh(prev_field)))

        # sum_int2 = sum(log(cosh(cur_field)))
        sum_int2 = np.sum(np.log(np.cosh(cur_field)))

        return (summation - sum_int1 - sum_int2)[0,0]

def _jacobian_sync(W, gain, v):
        """ 
        Computes the Jacobian of f at v, where f(v)=tanh(gain*Wv)
        f(v)[i] = tanh(gain*W[i,:].v)
        So df[i]/dv[j] = tanh'(gain*W[i,:].v)*gain*W[i,j]
        where tanh' = sech^2 = 1/cosh^2
        """

        res = gain/np.cosh(gain*np.matmul(W, v))**2
        J = np.matmul(np.diag(res), W)
        return J

def _jacobian_sync_subtract_I(W, gain, v):
        """ 
        Computes the Jacobian of f at v, where f(v)=tanh(gain*Wv)
        f(v)[i] = tanh(gain*W[i,:].v)
        So df[i]/dv[j] = tanh'(gain*W[i,:].v)*gain*W[i,j]
        where tanh' = sech^2 = 1/cosh^2
        """

        res = gain/np.cosh(gain*np.matmul(W, v))**2
        J = np.matmul(np.diag(res), W)
        return J-np.eye(W.shape[0])

def _update_async_stochastic(W, gain, a):
        """Update the current activation"""

        new_a = np.copy(a)
        for i in np.random.permutation(W.shape[0]):
            new_a[i] = np.tanh(gain*np.dot(W[i,:], new_a))
        return new_a

def _update_async_deterministic(W, gain, a):
        """Update the current activation"""

        new_a = np.copy(a)
        for i in xrange(W.shape[0]):
            new_a[i] = np.tanh(gain*np.dot(W[i,:], new_a))
        return new_a

def _energy_async(W, gain, cur_act, prev_act):
        """
        Uses eq. 4.3 in Soulie et al. 1989 with b_i=0
        But modifies eq. 4.2 to include gain, and adds gain term to first sum in eq. 4.3 (equivalent to just multiplying the weights by the gain before use).
        """

        # summation = -0.5 * cur_act^T * W * cur_act
        summation = -0.5*gain*np.matmul(cur_act[None,:], np.matmul(W, cur_act[:,None]))

        # integral(arctanh(x)) = 0.5*log(1-x^2) + x*arctanh(x); note that this equals 0 at 0

        sum_int = np.sum(0.5*np.log(1 - cur_act**2) + cur_act * np.arctanh(cur_act))

        return (summation + sum_int)[0,0]

def _jacobian_async_deterministic(W, gain, v):
        """ 
        Computes the Jacobian of f at v, where f(v)=tanh(gain*Wv).
        See Garrett's notes for derivation.
        TODO: optimize
        """

        def partial_intermediate_partial_intermediate(v_prev,i,j,k):
            if i == j:
                return gain*W[i,k]/np.cosh(gain*np.dot(W[i,:],v_prev))**2
            elif j == k:
                return 1.0
            else:
                return 0.0

        n = W.shape[0]
        intermediate_v = [v]
        # Assume stochastic=False
        for i in xrange(n):
            prev = np.copy(intermediate_v[i])
            prev[i] = np.tanh(gain*np.dot(W[i,:], prev))
            intermediate_v.append(prev)
        # Note: intermediate_v is shifted: intermediate_v[i] = v^(i-1) in notes

        prev_M_prod = np.zeros((n,n))

        # M^(0)
        for j in xrange(n):
            for l in xrange(n):
                prev_M_prod[j,l] = partial_intermediate_partial_intermediate(v, 0, j, l)


        for i in xrange(1,n+1):
            cur_M_prod = np.zeros((n,n))

            for j in xrange(n):
                for l in xrange(n):
                    for k in xrange(n):
                        cur_M_prod[j,l] += partial_intermediate_partial_intermediate(intermediate_v[i], i, j, k)*prev_M_prod[k,l]
            prev_M_prod = cur_M_prod

        return prev_M_prod

def _jacobian_async_deterministic_subtract_I(W, gain, v):
        """ 
        Computes the Jacobian of f at v, where f(v)=tanh(gain*Wv).
        See Garrett's notes for derivation.
        TODO: optimize
        """

        def partial_intermediate_partial_intermediate(v_prev,i,j,k):
            if i == j:
                return gain*W[i,k]/np.cosh(gain*np.dot(W[i,:],v_prev))**2
            elif j == k:
                return 1.0
            else:
                return 0.0

        n = W.shape[0]
        intermediate_v = [v]
        # Assume stochastic=False
        for i in xrange(n):
            prev = np.copy(intermediate_v[i])
            prev[i] = np.tanh(gain*np.dot(W[i,:], prev))
            intermediate_v.append(prev)
        # Note: intermediate_v is shifted: intermediate_v[i] = v^(i-1) in notes

        prev_M_prod = np.zeros((n,n))

        # M^(0)
        for j in xrange(n):
            for l in xrange(n):
                prev_M_prod[j,l] = partial_intermediate_partial_intermediate(v, 0, j, l)


        for i in xrange(1,n+1):
            cur_M_prod = np.zeros((n,n))

            for j in xrange(n):
                for l in xrange(n):
                    for k in xrange(n):
                        cur_M_prod[j,l] += partial_intermediate_partial_intermediate(intermediate_v[i], i, j, k)*prev_M_prod[k,l]
            prev_M_prod = cur_M_prod

        return prev_M_prod-np.eye(n)

def _no_energy(W,g,a,p):
    """ Energy function to use if one isn't defined """

    return float('nan')

def _no_jacobian(W,g,a):
    """ Jacobian function to use if one isn't defined """

    raise NotImplementedError

def _hopfield_energy(W, gain, cur_act, prev_act):
    """ Calculate the energy using Hopfield's derivation. """
    # TODO: I think this equals 0.5(Wv).v --> check this
    n = W.shape[0]
    e = 0
    for i in xrange(n):
        for j in xrange(n):
            e -= gain*W[i,j]*cur_act[i]*cur_act[j]
    return 0.5*e



class _Hopnet:
    def __init__(self, n, update_fun, energy_fun, jacobian_fun, gain=1.0):
        """Initialize a customized Hopnet"""

        self.n = n
        self.a = np.zeros(n, dtype=np.float32)
        self.W = np.zeros((n,n), dtype=np.float32)
        self.e = 0
        self.gain = gain
        self.update_fun = update_fun
        self.energy_fun = energy_fun
        self.jacobian_fun = jacobian_fun

    def learn(self, data):
        """Learn the data using Hebbian learning"""

        self.W = np.matmul(data, data.T)
        for i in range(self.n):
            self.W[i,i] = 0
        self.W *= (1.0/self.n)

    def update(self):
        return self.update_fun(self.W, self.gain, self.a)


    def simhop(self, a_init, tolerance=1e-05, max_steps=500, silent=False, fileid=sys.stdout):
        """Simulate the Hopnet until termination conditions are reached"""

        if len(a_init.shape) == 1:
            a_init = a_init.flatten()
        if len(a_init) != self.n:
            raise ValueError('The given a_init does not have {} entries.'.format(self.n))

        self.a = np.copy(a_init)

        t = 0
        cont = True

        if silent: # More optimized mode without any output
            while cont and t < max_steps:
                prev_a = np.copy(self.a)
                self.a = self.update()
                cont = not np.allclose(prev_a, self.a, rtol=0, atol=tolerance)
                t += 1
            # Update the energy only at the very end
            self.e = self.energy(self.a, prev_a)
        else: # Slower mode that provides output at each step
            self.e = self.energy(self.a, None)
            while cont and t < max_steps:
                self.show_state(t, fileid=fileid)

                prev_a = np.copy(self.a)
                self.a = self.update()
                self.e = self.energy(self.a, prev_a)
                cont = not np.allclose(prev_a, self.a, rtol=0, atol=tolerance)
                t += 1
            # Show final state
            self.show_state(t, fileid=fileid)
        return t, self.a, self.e

    def energy(self, cur_act, prev_act):
        return self.energy_fun(self.W, self.gain, cur_act, prev_act)

    def jacobian(self, v):
        return self.jacobian_fun(self.W, self.gain, v)

    def show_state(self, t, fileid=sys.stdout):
        """Print the current state"""

        fileid.write("t:{:4d} [".format(t))
        for i in self.a:
            fileid.write(' {0: 5.3f}'.format(i))
        fileid.write(" ]  E: {}\n".format(self.e))

    def show_wts(self, fileid=sys.stdout):
        """Print the weight matrix"""

        fileid.write("\nWeights =\n")
        for i in range(self.n):
            for j in range(self.n):
                fileid.write(" {0: 7.3f}".format(self.W[i,j]))
            fileid.write("\n")






modes = {
    'sync': (_update_sync, _energy_sync,_jacobian_sync),
    'async_stochastic': (_update_async_stochastic, _energy_async, _no_jacobian),
    'async_deterministic': (_update_async_deterministic, _energy_async, _jacobian_async_deterministic)
}


def Hopnet(n, mode=modes['sync'], gain=1.0):
    """
    Factory function for creating Hopnets.
    A mode is defined as a 3-tuple of functions (update,energy,Jacobian).
    The last two can be None.
    An update function is defined by W,gain,a-->a
    An energy function is defined by W,gain,a,prev_a-->e
    A Jacobian function is defined by W,gain,a-->J
    """

    mode_lst = list(mode)
    if mode_lst[0] is None:
        raise ValueError('Invalid mode.')
    if mode_lst[1] is None:
        mode_lst[1] = _no_energy
    if mode_lst[2] is None:
        mode_lst[2] = _no_jacobian

    return _Hopnet(n, *mode_lst, gain=gain)