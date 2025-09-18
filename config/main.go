package main

import (
	"fmt"

	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/engine"
)

func renderChart() {
	chartPath := "./charts/inferencepool" // Path to your Helm chart
	chart, err := loader.Load(chartPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to load chart: %v", err))
	}
	values, _ := chartutil.ReadValuesFile("charts/inferencepool/values.yaml")
	options := chartutil.ReleaseOptions{
		Name:      "test-release",
		Namespace: "default",
	}
	renderValues, err := chartutil.ToRenderValues(chart, values, options, nil)
	if err != nil {
		panic(fmt.Sprintf("Failed to create render values: %v", err))
	}
	fmt.Println(values)
	rendered, err := engine.Render(chart, renderValues)
	if err != nil {
		panic(fmt.Sprintf("Failed to render chart: %v", err))
	}

	// Assert on the content of the rendered manifests
	for name, content := range rendered {
		fmt.Printf("--- Rendered File: %s ---\n%s\n", name, content)
	}
}

func main() {
	renderChart()
}
