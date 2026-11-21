package hook_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleClipGrad() {
	p := variable.New(1)
	p.Grad = variable.New(1, 2, 3, 4)

	h := hook.ClipGrad(1.0)
	h(layer.Parameters{"p": p})

	fmt.Println(p.Grad)

	// Output:
	// variable[4]([0.18257415 0.3651483 0.54772246 0.7302966])
}

func ExampleClipGrad_noclip() {
	p := variable.New(1)
	p.Grad = variable.New(0.1, 0.2, 0.3, 0.4)

	h := hook.ClipGrad(1.0)
	h(layer.Parameters{"p": p})

	fmt.Println(p.Grad)

	// Output:
	// variable[4]([0.1 0.2 0.3 0.4])
}

func ExampleClipGrad_nograd() {
	p0 := variable.New(1)
	p1 := variable.New(2)
	p1.Grad = variable.New(0.1, 0.2, 0.3, 0.4)

	h := hook.ClipGrad(0.1)
	h(layer.Parameters{"p0": p0, "p1": p1})

	fmt.Println(p0.Grad)
	fmt.Println(p1.Grad)

	// Output:
	// <nil>
	// variable[4]([0.018257385 0.03651477 0.054772153 0.07302954])
}

func ExampleClipGrad_sharedGrad() {
	x0, x1 := variable.New(1), variable.New(2)

	y := variable.Add(x0, x1)
	y.Backward()

	h := hook.ClipGrad(1.0)
	h(layer.Parameters{
		"x0": x0,
		"x1": x1,
	})

	fmt.Printf("%.6f\n", x0.Grad.At())
	fmt.Printf("%.6f\n", x1.Grad.At())

	// Output:
	// 0.707106
	// 0.707106
}

func TestClipGrad_sharedGrad(t *testing.T) {
	cases := []struct {
		name      string
		gradients [][]float32
		sharing   []int
		max       float32
		want      [][]float32
	}{
		{
			name:      "two parameters share a gradient",
			gradients: [][]float32{{1}},
			sharing:   []int{0, 0},
			max:       1,
			want:      [][]float32{{0.707106}, {0.707106}},
		},
		{
			name:      "three parameters share a gradient",
			gradients: [][]float32{{1}},
			sharing:   []int{0, 0, 0},
			max:       1,
			want:      [][]float32{{0.577350}, {0.577350}, {0.577350}},
		},
		{
			name:      "shared vector gradient",
			gradients: [][]float32{{3, 4}},
			sharing:   []int{0, 0},
			max:       1,
			want:      [][]float32{{0.424264, 0.565685}, {0.424264, 0.565685}},
		},
		{
			name:      "mixed shared and independent gradients",
			gradients: [][]float32{{1}, {2}},
			sharing:   []int{0, 0, 1},
			max:       1,
			want:      [][]float32{{0.408248}, {0.408248}, {0.816496}},
		},
		{
			name:      "independent equal gradients",
			gradients: [][]float32{{1}, {1}},
			sharing:   []int{0, 1},
			max:       1,
			want:      [][]float32{{0.707106}, {0.707106}},
		},
		{
			name:      "shared gradient below clipping threshold",
			gradients: [][]float32{{1}},
			sharing:   []int{0, 0},
			max:       2,
			want:      [][]float32{{1}, {1}},
		},
		{
			name:      "zero clipping threshold",
			gradients: [][]float32{{1}},
			sharing:   []int{0, 0},
			max:       0,
			want:      [][]float32{{0}, {0}},
		},
		{
			name:      "parameter without a gradient",
			gradients: [][]float32{{1}},
			sharing:   []int{0, 0, -1},
			max:       1,
			want:      [][]float32{{0.707106}, {0.707106}, nil},
		},
	}

	for _, c := range cases {
		gradients := make([]*variable.Variable, len(c.gradients))
		for i, data := range c.gradients {
			gradients[i] = variable.New(data...)
		}

		params := make(layer.Parameters)
		for i, group := range c.sharing {
			p := variable.New(1)
			if group >= 0 {
				p = variable.Zeros(len(c.gradients[group]))
				p.Grad = gradients[group]
			}

			params[fmt.Sprint(i)] = p
		}

		h := hook.ClipGrad(c.max)
		h(params)

		for i, want := range c.want {
			p := params[fmt.Sprint(i)]
			if want == nil {
				if p.Grad != nil {
					t.Errorf("%s: parameter %d gradient = %v, want nil", c.name, i, p.Grad)
				}
				continue
			}

			if got := p.Grad; got == nil || !tensor.SliceEqual(got.Shape(), []int{len(want)}) || !tensor.IsCloseAll(got.Data, variable.New(want...).Data) {
				t.Errorf("%s: parameter %d gradient = %v, want %v", c.name, i, got, want)
			}
		}
	}
}
