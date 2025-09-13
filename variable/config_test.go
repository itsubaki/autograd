package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleNograd() {
	f := func() {
		x := variable.New(3)
		y := variable.Square(x)
		y.Backward()

		fmt.Println("gx: ", x.Grad)
		fmt.Println()
	}

	fmt.Println("backprop:", variable.Config.EnableBackprop)
	f()

	func() {
		defer variable.Nograd().End()

		fmt.Println("backprop:", variable.Config.EnableBackprop)
		f()
	}()

	fmt.Println("backprop:", variable.Config.EnableBackprop)
	f()

	// Output:
	// backprop: true
	// gx:  variable(6)
	//
	// backprop: false
	// gx:  <nil>
	//
	// backprop: true
	// gx:  variable(6)
}

func ExampleTestMode() {
	fmt.Println("train:", variable.Config.Train)

	func() {
		defer variable.TestMode().End()

		fmt.Println("train:", variable.Config.Train)
	}()

	fmt.Println("train:", variable.Config.Train)

	// Output:
	// train: true
	// train: false
	// train: true
}
