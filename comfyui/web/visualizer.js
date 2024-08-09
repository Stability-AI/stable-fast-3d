import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const container = document.getElementById("container");
const visualizer = document.getElementById("visualizer");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setClearColor(0x808080);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);

const controls = new OrbitControls(camera, renderer.domElement);
controls.dampingFactor = 0.25;
controls.enableDamping = true;
controls.enableZoom = true;

const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
directionalLight.position.set(0.5, 1, -1.5);
const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.5);


var lastTimestamp = "";

window.onresize = function () {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
};

function render() {
    var timestamp = visualizer.getAttribute("timestamp");
    var b64_glb = visualizer.getAttribute("b64_glb");
    if (timestamp != lastTimestamp) {
        lastTimestamp = timestamp;
        init(b64_glb);
    }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(render);
}

async function init(b64_glb) {
    scene.clear();
    scene.add(ambientLight);
    scene.add(camera);
    scene.add(directionalLight);
    scene.add(hemisphereLight);

    if (b64_glb) {
        const loader = new GLTFLoader();
        const glbData = atob(b64_glb);
        const glbBuffer = new Uint8Array(glbData.length);
        for (let i = 0; i < glbData.length; i++) {
            glbBuffer[i] = glbData.charCodeAt(i);
        }

        loader.parse(glbBuffer.buffer, '', (gltf) => {
            scene.add(gltf.scene);

            const box = new THREE.Box3().setFromObject(gltf.scene);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);

            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            camera.position.z = cameraZ * -1.5;
            camera.lookAt(center);

            controls.target.copy(center);
            controls.update();
        }, undefined, (error) => {
            console.error('An error occurred loading GLB:', error);
        });
    }

    render();
}

init();
