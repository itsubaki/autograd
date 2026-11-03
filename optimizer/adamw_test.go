package optimizer_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleAdamW() {
	p := variable.New(1.0)
	p.Grad = variable.New(1.0)
	m := &TestModel{P: p}

	optim := &optimizer.AdamW{
		Alpha:       0.001,
		Beta1:       0.9,
		Beta2:       0.999,
		WeightDecay: 0.1,
	}

	for range 2 {
		optim.Update(m.Params())
		fmt.Println(p)
	}

	// Output:
	// variable(0.9989)
	// variable(0.9978001)
}

func ExampleAdamW_nograd() {
	p := variable.New(1.0)
	m := &TestModel{P: p}

	optim := &optimizer.AdamW{
		Alpha:       0.001,
		Beta1:       0.9,
		Beta2:       0.999,
		WeightDecay: 0.1,
	}

	optim.Update(m.Params())
	fmt.Println(p)

	// Output:
	// variable(1)
}

func TestAdamW(t *testing.T) {
	cases := []struct {
		name  string
		grad  float32
		alpha float32
		decay float32
		want  []float32
	}{
		{
			name: "zero gradient",
			grad: 0, alpha: 0.001, decay: 0.1,
			want: []float32{0.9999, 0.99980001},
		},
		{
			name: "decay before gradient update",
			grad: 1, alpha: 0.1, decay: 0.2,
			want: []float32{0.88, 0.7624},
		},
	}

	for _, c := range cases {
		p := variable.New(1)
		p.Grad = variable.New(c.grad)

		m := &TestModel{P: p}
		optim := &optimizer.AdamW{
			Alpha: c.alpha, Beta1: 0.9, Beta2: 0.999, WeightDecay: c.decay,
		}

		for i, want := range c.want {
			optim.Update(m.Params())
			if got := p.At(); math.Abs(float64(got-want)) > 1e-6 {
				t.Errorf("step %d: got=%v, want=%v", i+1, got, want)
			}
		}
	}
}
