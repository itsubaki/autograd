package variable

import "fmt"

type Forwarder interface {
	Forward(x ...*Variable) []*Variable
	Backward(gy ...*Variable) []*Variable
}

type Function struct {
	Input, Output []*Variable
	Generation    int
	Forwarder
}

// ApplyAndFirst applies the function and returns the first output
func (f *Function) ApplyAndFirst(x ...*Variable) *Variable {
	return f.Apply(x...)[0]
}

// Apply applies the function
func (f *Function) Apply(x ...*Variable) []*Variable {
	y := f.Forward(x...)

	if Config.EnableBackprop {
		f.Generation = maxgen(x...)
		f.setCreator(y)
		f.Input, f.Output = x, y
	}

	return y
}

func (f Function) String() string {
	return fmt.Sprintf("%T%v", f.Forwarder, f.Input)
}

func (f *Function) setCreator(y []*Variable) {
	for i := range y {
		y[i].SetCreator(f)
	}
}

func maxgen(x ...*Variable) int {
	var max int
	for _, v := range x {
		if max < v.Generation {
			max = v.Generation
		}
	}

	return max
}
