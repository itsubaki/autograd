package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleNograd() {
	fmt.Println("enable_backprop:", variable.Config.EnableBackprop)

	func() {
		defer variable.Nograd()()
		fmt.Println("enable_backprop:", variable.Config.EnableBackprop)

		x := variable.New(3)
		y := variable.Square(x)
		y.Backward()
		fmt.Println("gx: ", x.Grad)
	}()

	fmt.Println("enable_backprop:", variable.Config.EnableBackprop)

	x := variable.New(3)
	y := variable.Square(x)
	y.Backward()
	fmt.Println("gx: ", x.Grad)

	// Output:
	// enable_backprop: true
	// enable_backprop: false
	// gx:  <nil>
	// enable_backprop: true
	// gx:  variable([6])
}

func ExampleTestMode() {
	fmt.Println("train:", variable.Config.Train)

	func() {
		defer variable.TestMode()()
		fmt.Println("train:", variable.Config.Train)
	}()

	fmt.Println("train:", variable.Config.Train)

	// Output:
	// train: true
	// train: false
	// train: true
}
