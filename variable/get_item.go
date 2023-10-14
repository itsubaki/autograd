package variable

func GetItem(slices []int) func(x ...*Variable) *Variable {
	return (&Function{Forwarder: &GetItemT{Slices: slices}}).First
}

type GetItemT struct {
	Slices []int
	xShape []int
}

func (f *GetItemT) Forward(x ...*Variable) []*Variable {
	f.xShape = Shape(x[0])

	y := make([][]float64, len(f.Slices))
	for i, idx := range f.Slices {
		y[i] = x[0].Data[idx]
	}

	return []*Variable{
		NewOf(y...),
	}
}

func (f *GetItemT) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		GetItemGrad(f.Slices, f.xShape)(gy...),
	}
}
