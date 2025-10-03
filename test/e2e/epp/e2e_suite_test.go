/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package epp

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"

	infextv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	infextv1a2 "sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"

	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/engine"
)

const (
	// defaultExistsTimeout is the default timeout for a resource to exist in the api server.
	defaultExistsTimeout = 30 * time.Second
	// defaultReadyTimeout is the default timeout for a resource to report a ready state.
	defaultReadyTimeout = 3 * time.Minute
	// defaultModelReadyTimeout is the default timeout for the model server deployment to report a ready state.
	defaultModelReadyTimeout = 10 * time.Minute
	// defaultCurlTimeout is the default timeout for the curl command to get a response.
	defaultCurlTimeout = 30 * time.Second
	// defaultInterval is the default interval to check if a resource exists or ready conditions.
	defaultInterval = time.Millisecond * 250
	// defaultCurlInterval is the default interval to run the test curl command.
	defaultCurlInterval = time.Second * 5
	// defaultNsName is the default name of the Namespace used for tests. Can override using the E2E_NS environment variable.
	defaultNsName = "inf-ext-e2e"
	// modelServerName is the name of the model server test resources.
	modelServerName = "vllm-llama3-8b-instruct"
	// modelName is the test model name.
	modelName = "food-review"
	// targetModelName is the target model name of the test model server.
	targetModelName = modelName + "-1"
	// envoyName is the name of the envoy proxy test resources.
	envoyName = "envoy"
	// envoyPort is the listener port number of the test envoy proxy.
	envoyPort = "8081"
	// inferExtName is the name of the inference extension test resources.
	inferExtName = "vllm-llama3-8b-instruct-epp"
	// metricsReaderSecretName is the name of the metrics reader secret which stores sa token to read epp metrics.
	metricsReaderSecretName = "inference-gateway-sa-metrics-reader-secret"
	// clientManifest is the manifest for the client test resources.
	clientManifest = "../../testdata/client.yaml"
	// modelServerSecretManifest is the manifest for the model server secret resource.
	modelServerSecretManifest = "../../testdata/model-secret.yaml"
	// xInferPoolManifest is the manifest for the inference pool CRD with 'inference.networking.x-k8s.io' group.
	xInferPoolManifest = "../../../config/crd/bases/inference.networking.x-k8s.io_inferencepools.yaml"
	// xInferObjectiveManifest is the manifest for the inference model CRD with 'inference.networking.x-k8s.io' group.
	xInferObjectiveManifest = "../../../config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml"
	// inferPoolManifest is the manifest for the inference pool CRD with 'inference.networking.k8s.io' group.
	inferPoolManifest = "../../../config/crd/bases/inference.networking.k8s.io_inferencepools.yaml"
	// inferExtManifestDefault is the manifest for the default inference extension test resources (single replica).
	inferExtManifestDefault = "../../testdata/inferencepool-e2e.yaml"
	// inferExtManifestLeaderElection is the manifest for the inference extension test resources with leader election enabled (3 replicas).
	inferExtManifestLeaderElection = "../../testdata/inferencepool-leader-election-e2e.yaml"
	// envoyManifest is the manifest for the envoy proxy test resources.
	envoyManifest = "../../testdata/envoy.yaml"
	// metricsRbacManifest is the manifest for the rbac resources for testing metrics.
	metricsRbacManifest = "../../testdata/metrics-rbac.yaml"
	// modelServerManifestFilepathEnvVar is the env var that holds absolute path to the manifest for the model server test resource.
	modelServerManifestFilepathEnvVar = "MANIFEST_PATH"
	// replicaCount is the number of replicas of EPP.
	replicaCount = 3

	name = "vllm-llama3-8b-instruct"
)

const e2eLeaderElectionEnabledEnvVar = "E2E_LEADER_ELECTION_ENABLED"

var (
	ctx = context.Background()
	cli client.Client
	// Required for exec'ing in curl pod
	kubeCli               *kubernetes.Clientset
	scheme                = runtime.NewScheme()
	cfg                   = config.GetConfigOrDie()
	nsName                string
	e2eImage              string
	leaderElectionEnabled bool
)

func TestAPIs(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t,
		"End To End Test Suite",
	)
}

func renderChartsToYamls(nsName string) []string {
	chartPath := "/usr/local/google/home/rahulgurnani/gateway-api-inference-extension/config/charts/inferencepool"
	chart, err := loader.Load(chartPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to load chart: %v", err))
	}
	values, _ := chartutil.ReadValuesFile("/usr/local/google/home/rahulgurnani/gateway-api-inference-extension/config/charts/inferencepool/values.yaml")
	infExt, ok := values["inferenceExtension"].(map[string]interface{})
	if ok {
		infExt["replicas"] = replicaCount
		fmt.Println(infExt)
		flags, ok := infExt["flags"].([]interface{})
		if ok {
			flags = append(flags, map[string]string{
				"name":  "ha-enable-leader-election",
				"value": "true",
			})
			infExt["flags"] = flags
		}
	}

	options := chartutil.ReleaseOptions{
		Name:      name,
		Namespace: nsName,
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
	var renderedValues []string
	for fName, renderedChart := range rendered {
		if strings.Contains(fName, "NOTES.txt") {
			continue
		}

		fmt.Println("----------------rendered----------------")
		fmt.Println(fName)
		objs := strings.Split(renderedChart, "\n---")
		for _, obj := range objs {
			fmt.Println("-----------obj-----------")
			fmt.Println(obj)
			renderedValues = append(renderedValues, obj)
		}
	}

	return renderedValues
}

var _ = ginkgo.BeforeSuite(func() {
	nsName = os.Getenv("E2E_NS")
	if nsName == "" {
		nsName = defaultNsName
	}
	e2eImage = os.Getenv("E2E_IMAGE")
	gomega.Expect(e2eImage).NotTo(gomega.BeEmpty(), "E2E_IMAGE environment variable is not set")

	if os.Getenv(e2eLeaderElectionEnabledEnvVar) == "true" {
		leaderElectionEnabled = true
		ginkgo.By("Leader election test mode enabled via " + e2eLeaderElectionEnabledEnvVar)
	}
	leaderElectionEnabled = true

	ginkgo.By("Setting up the test suite")
	setupSuite()

	ginkgo.By("Creating test infrastructure")
	setupInfra()
})

func setupInfra() {
	// this function ensures ModelServer manifest path exists.
	// run this before createNs to fail fast in case it doesn't.
	modelServerManifestPath := readModelServerManifestPath()

	createNamespace(cli, nsName)

	modelServerManifestArray := getYamlsFromModelServerManifest(modelServerManifestPath)
	if strings.Contains(modelServerManifestArray[0], "hf-token") {
		createHfSecret(cli, modelServerSecretManifest)
	}
	crds := map[string]string{
		"inferencepools.inference.networking.x-k8s.io":      xInferPoolManifest,
		"inferencepools.inference.networking.k8s.io":        inferPoolManifest,
		"inferenceobjectives.inference.networking.x-k8s.io": xInferObjectiveManifest,
	}

	createCRDs(cli, crds)

	createInferExt(cli)
	createClient(cli, clientManifest)
	createEnvoy(cli, envoyManifest)
	createMetricsRbac(cli, metricsRbacManifest)
	// Run this step last, as it requires additional time for the model server to become ready.
	ginkgo.By("Creating model server resources from manifest: " + modelServerManifestPath)
	createModelServer(cli, modelServerManifestArray)
}

var _ = ginkgo.AfterSuite(func() {
	// If E2E_PAUSE_ON_EXIT is set, pause the test run before cleanup.
	// This is useful for debugging the state of the cluster after the test has run.
	if pauseStr := os.Getenv("E2E_PAUSE_ON_EXIT"); pauseStr != "" {
		ginkgo.By("Pausing before cleanup as requested by E2E_PAUSE_ON_EXIT=" + pauseStr)
		pauseDuration, err := time.ParseDuration(pauseStr)
		if err != nil {
			// If it's not a valid duration (e.g., "true"), just wait indefinitely.
			ginkgo.By("Invalid duration, pausing indefinitely. Press Ctrl+C to stop the test runner when you are done.")
			select {} // Block forever
		}
		ginkgo.By(fmt.Sprintf("Pausing for %v...", pauseDuration))
		time.Sleep(pauseDuration)
	}

	ginkgo.By("Performing global cleanup")
	cleanupResources()
})

// setupSuite initializes the test suite by setting up the Kubernetes client,
// loading required API schemes, and validating configuration.
func setupSuite() {
	gomega.ExpectWithOffset(1, cfg).NotTo(gomega.BeNil())

	err := clientgoscheme.AddToScheme(scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	err = apiextv1.AddToScheme(scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	// TODO: Fix the v1a2 chart
	err = infextv1a2.Install(scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	err = infextv1.Install(scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	cli, err = client.New(cfg, client.Options{Scheme: scheme})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(cli).NotTo(gomega.BeNil())

	kubeCli, err = kubernetes.NewForConfig(cfg)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(kubeCli).NotTo(gomega.BeNil())
}

func cleanupResources() {
	if cli == nil {
		return // could happen if BeforeSuite had an error
	}

	gomega.Expect(testutils.DeleteClusterResources(ctx, cli)).To(gomega.Succeed())
	gomega.Expect(testutils.DeleteNamespacedResources(ctx, cli, nsName)).To(gomega.Succeed())
}

func cleanupInferModelResources() {
	gomega.Expect(testutils.DeleteInferenceObjectiveResources(ctx, cli, nsName)).To(gomega.Succeed())
}

func getTimeout(key string, fallback time.Duration) time.Duration {
	if value, ok := os.LookupEnv(key); ok {
		if parsed, err := time.ParseDuration(value); err == nil {
			return parsed
		}
	}
	return fallback
}

var (
	existsTimeout     = getTimeout("EXISTS_TIMEOUT", defaultExistsTimeout)
	readyTimeout      = getTimeout("READY_TIMEOUT", defaultReadyTimeout)
	modelReadyTimeout = getTimeout("MODEL_READY_TIMEOUT", defaultModelReadyTimeout)
	curlTimeout       = getTimeout("CURL_TIMEOUT", defaultCurlTimeout)
	interval          = defaultInterval
	curlInterval      = defaultCurlInterval
)

func createNamespace(k8sClient client.Client, ns string) {
	ginkgo.By("Creating e2e namespace: " + ns)
	obj := &corev1.Namespace{
		ObjectMeta: v1.ObjectMeta{
			Name: ns,
		},
	}
	err := k8sClient.Create(ctx, obj)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "Failed to create e2e test namespace")
}

// namespaceExists ensures that a specified namespace exists and is ready for use.
func namespaceExists(k8sClient client.Client, ns string) {
	ginkgo.By("Ensuring namespace exists: " + ns)
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: ns}, &corev1.Namespace{})
	}, existsTimeout, interval)
}

// readModelServerManifestPath reads from env var the absolute filepath to model server deployment for testing.
func readModelServerManifestPath() string {
	ginkgo.By(fmt.Sprintf("Ensuring %s environment variable is set", modelServerManifestFilepathEnvVar))
	modelServerManifestFilepath := os.Getenv(modelServerManifestFilepathEnvVar)
	gomega.Expect(modelServerManifestFilepath).NotTo(gomega.BeEmpty(), modelServerManifestFilepathEnvVar+" is not set")
	return modelServerManifestFilepath
}

func getYamlsFromModelServerManifest(modelServerManifestPath string) []string {
	ginkgo.By("Ensuring the model server manifest points to an existing file")
	modelServerManifestArray := readYaml(modelServerManifestPath)
	gomega.Expect(modelServerManifestArray).NotTo(gomega.BeEmpty())
	return modelServerManifestArray
}

// createCRDs creates the Inference Extension CRDs used for testing.
func createCRDs(k8sClient client.Client, crds map[string]string) {
	for name, path := range crds {
		ginkgo.By("Creating CRD resource from manifest: " + path)
		applyYAMLFile(k8sClient, path)

		// Wait for the CRD to exist.
		crd := &apiextv1.CustomResourceDefinition{}
		testutils.EventuallyExists(ctx, func() error {
			return k8sClient.Get(ctx, types.NamespacedName{Name: name}, crd)
		}, existsTimeout, interval)

		// Wait for the CRD to be established.
		testutils.CRDEstablished(ctx, k8sClient, crd, readyTimeout, interval)
	}
}

// createClient creates the client pod used for testing from the given filePath.
func createClient(k8sClient client.Client, filePath string) {
	ginkgo.By("Creating client resources from manifest: " + filePath)
	applyYAMLFile(k8sClient, filePath)

	// Wait for the pod to exist.
	pod := &corev1.Pod{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: "curl"}, pod)
	}, existsTimeout, interval)

	// Wait for the pod to be ready.
	testutils.PodReady(ctx, k8sClient, pod, readyTimeout, interval)
}

// createMetricsRbac creates the metrics RBAC resources from the manifest file.
func createMetricsRbac(k8sClient client.Client, filePath string) {
	inManifests := readYaml(filePath)
	ginkgo.By("Replacing placeholder namespace with E2E_NS environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.ReplaceAll(m, "$E2E_NS", nsName))
	}
	ginkgo.By("Creating RBAC resources for scraping metrics from manifest: " + filePath)
	createObjsFromYaml(k8sClient, outManifests)

	// wait for sa token to exist
	testutils.EventuallyExists(ctx, func() error {
		token, err := getMetricsReaderToken(k8sClient)
		if err != nil {
			return err
		}
		if len(token) == 0 {
			return errors.New("failed to get metrics reader token")
		}
		return nil
	}, existsTimeout, interval)
}

// createModelServer creates the model server resources used for testing from the given filePaths.
func createModelServer(k8sClient client.Client, modelServerManifestArray []string) {
	createObjsFromYaml(k8sClient, modelServerManifestArray)

	// Wait for the deployment to exist.
	deploy := &appsv1.Deployment{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: modelServerName}, deploy)
	}, existsTimeout, interval)

	// Wait for the deployment to be available.
	testutils.DeploymentAvailable(ctx, k8sClient, deploy, modelReadyTimeout, interval)
}

// createHfSecret read HF_TOKEN from env var and creates a secret that contains the access token.
func createHfSecret(k8sClient client.Client, secretPath string) {
	ginkgo.By("Ensuring the HF_TOKEN environment variable is set")
	token := os.Getenv("HF_TOKEN")
	gomega.Expect(token).NotTo(gomega.BeEmpty(), "HF_TOKEN is not set")

	inManifests := readYaml(secretPath)
	ginkgo.By("Replacing placeholder secret data with HF_TOKEN environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.Replace(m, "$HF_TOKEN", token, 1))
	}

	ginkgo.By("Creating model server secret resource")
	createObjsFromYaml(k8sClient, outManifests)

	// Wait for the secret to exist before proceeding with test.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: "hf-token"}, &corev1.Secret{})
	}, existsTimeout, interval)
}

// createEnvoy creates the envoy proxy resources used for testing from the given filePath.
func createEnvoy(k8sClient client.Client, filePath string) {
	inManifests := readYaml(filePath)
	ginkgo.By("Replacing placeholder namespace with E2E_NS environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.ReplaceAll(m, "$E2E_NS", nsName))
	}

	ginkgo.By("Creating envoy proxy resources from manifest: " + filePath)
	createObjsFromYaml(k8sClient, outManifests)

	// Wait for the configmap to exist before proceeding with test.
	cfgMap := &corev1.ConfigMap{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: envoyName}, cfgMap)
	}, existsTimeout, interval)

	// Wait for the deployment to exist.
	deploy := &appsv1.Deployment{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: envoyName}, deploy)
	}, existsTimeout, interval)

	// Wait for the deployment to be available.
	testutils.DeploymentAvailable(ctx, k8sClient, deploy, readyTimeout, interval)

	// Wait for the service to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: envoyName}, &corev1.Service{})
	}, existsTimeout, interval)
}

// createInferExt creates the inference extension resources used for testing from the given filePath.
func createInferExt(k8sClient client.Client) {
	outManifests := renderChartsToYamls(nsName)

	ginkgo.By("Creating inference extension resources from outManifests")
	createObjsFromYaml(k8sClient, outManifests)

	// Wait for the deployment to exist.
	deploy := &appsv1.Deployment{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: inferExtName}, deploy)
	}, existsTimeout, interval)

	if leaderElectionEnabled {
		// With leader election enabled, only 1 replica will be "Ready" at any given time (the leader).
		testutils.DeploymentReadyReplicas(ctx, k8sClient, deploy, 1, modelReadyTimeout, interval)
	} else {
		testutils.DeploymentAvailable(ctx, k8sClient, deploy, modelReadyTimeout, interval)
	}

	// Wait for the service to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: inferExtName}, &corev1.Service{})
	}, existsTimeout, interval)
}

// applyYAMLFile reads a file containing YAML (possibly multiple docs)
// and applies each object to the cluster.
func applyYAMLFile(k8sClient client.Client, filePath string) {
	// Create the resources from the manifest file
	createObjsFromYaml(k8sClient, readYaml(filePath))
}

func readYaml(filePath string) []string {
	ginkgo.By("Reading YAML file: " + filePath)
	yamlBytes, err := os.ReadFile(filePath)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Split multiple docs, if needed
	return strings.Split(string(yamlBytes), "\n---")
}

func createObjsFromYaml(k8sClient client.Client, docs []string) {
	// For each doc, decode and create
	decoder := serializer.NewCodecFactory(scheme).UniversalDeserializer()
	for _, doc := range docs {
		trimmed := strings.TrimSpace(doc)
		if trimmed == "" {
			continue
		}

		// Decode into a runtime.Object
		obj, gvk, decodeErr := decoder.Decode([]byte(trimmed), nil, nil)
		if decodeErr != nil {
			log.Printf("Trimmed: %s", trimmed)
			continue
		}
		gomega.Expect(decodeErr).NotTo(gomega.HaveOccurred(),
			"Failed to decode YAML document to a Kubernetes object")

		ginkgo.By(fmt.Sprintf("Decoded GVK: %s", gvk))

		unstrObj, ok := obj.(*unstructured.Unstructured)
		if !ok {
			// Fallback if it's a typed object
			unstrObj = &unstructured.Unstructured{}
			// Convert typed to unstructured
			err := scheme.Convert(obj, unstrObj, nil)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}

		unstrObj.SetNamespace(nsName)

		// Create the object
		err := k8sClient.Create(ctx, unstrObj)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(),
			"Failed to create object from YAML")
	}
}
