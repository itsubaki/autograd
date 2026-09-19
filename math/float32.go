package math

import "math"

var (
	Pi         = float32(math.Pi)
	Sqrt2      = float32(math.Sqrt2)
	MaxFloat32 = float32(math.MaxFloat32)
)

func Sin(x float32) float32 { return float32(math.Sin(float64(x))) }

func Cos(x float32) float32 { return float32(math.Cos(float64(x))) }

func Tanh(x float32) float32 { return float32(math.Tanh(float64(x))) }

func Exp(x float32) float32 { return float32(math.Exp(float64(x))) }

func Log(x float32) float32 { return float32(math.Log(float64(x))) }

func Sqrt(x float32) float32 { return float32(math.Sqrt(float64(x))) }

func Abs(x float32) float32 { return float32(math.Abs(float64(x))) }

func Inf(sign int) float32 { return float32(math.Inf(sign)) }

func NaN() float32 { return float32(math.NaN()) }

func Max(x, y float32) float32 { return float32(math.Max(float64(x), float64(y))) }

func Min(x, y float32) float32 { return float32(math.Min(float64(x), float64(y))) }

func Pow(x, y float32) float32 { return float32(math.Pow(float64(x), float64(y))) }
