package hook_test

import (
	"fmt"

	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleClipGrad() {
	p := variable.New(1)
	p.Grad = variable.New(1, 2, 3, 4)

	h := hook.ClipGrad(1.0)
	h(layer.Parameters{"p": p})

	fmt.Println(p.Grad)

	// Output:
	// variable[4]([0.18257415250172812 0.36514830500345624 0.5477224575051843 0.7302966100069125])
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
	// variable[4]([0.01825738525023306 0.03651477050046612 0.05477215575069918 0.07302954100093224])
}
