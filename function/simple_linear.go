package function

import "github.com/itsubaki/autograd/variable"

func LinearSimple(x, w *variable.Variable, b ...*variable.Variable) *variable.Variable {
	xw := MatMul(x, w)
	if len(b) == 0 {
		return xw
	}

	y := Add(xw, b[0])

	// `xw` is not used in backprop for `MatMul` and `Add`.
	// Clear to save memory.
	xw.Data = nil
	return y
}
