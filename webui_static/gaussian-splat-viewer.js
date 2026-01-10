/**
 * Gaussian Splat Viewer using Spark.js (THREE.js-based renderer)
 * https://github.com/sparkjsdev/spark
 *
 * Controls:
 * - WASD: Move forward/back/left/right
 * - Q/E: Move up/down
 * - Mouse drag: Look around
 * - Scroll: Adjust move speed
 */

class GaussianSplatViewer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.onProgress = options.onProgress || (() => {});
        this.onLoad = options.onLoad || (() => {});
        this.onError = options.onError || ((e) => console.error(e));

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.splat = null;
        this.animationId = null;

        // First-person controls state
        this.keys = {};
        this.mouseDown = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.yaw = Math.PI;    // Start facing toward -Z (toward the splat)
        this.pitch = 0;
        this.moveSpeed = 2.0;  // Units per second
        this.lookSpeed = 0.003;
        this.lastTime = performance.now();

        // Create a promise that resolves when init is complete
        this.ready = this.init();
    }

    async init() {
        try {
            console.log('GaussianSplatViewer: Starting initialization...');

            // Import THREE.js and Spark dynamically
            console.log('GaussianSplatViewer: Importing THREE.js...');
            const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js');

            console.log('GaussianSplatViewer: Importing Spark.js SplatMesh...');
            const { SplatMesh } = await import('https://sparkjs.dev/releases/spark/0.1.10/spark.module.js');

            this.THREE = THREE;
            this.SplatMesh = SplatMesh;

            console.log('GaussianSplatViewer: Setting up renderer...');

            // Setup renderer
            this.renderer = new THREE.WebGLRenderer({
                canvas: this.canvas,
                antialias: true,
                alpha: false
            });
            this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.setClearColor(0x000000, 1);

            // Setup scene
            this.scene = new THREE.Scene();

            // Setup camera
            this.camera = new THREE.PerspectiveCamera(
                60,
                this.canvas.clientWidth / this.canvas.clientHeight,
                0.01,
                1000
            );
            this.camera.position.set(0, 0, 3);

            // Setup first-person controls
            this.setupControls();

            // Handle resize
            this.resizeObserver = new ResizeObserver(() => this.handleResize());
            this.resizeObserver.observe(this.canvas);

            // Start render loop
            this.animate();

            console.log('GaussianSplatViewer: Initialization complete!');
        } catch (error) {
            console.error('GaussianSplatViewer: Failed to initialize:', error);
            this.onError(error);
            throw error;
        }
    }

    setupControls() {
        // Keyboard controls
        this.keyDownHandler = (e) => {
            // Only capture if canvas or body is focused (not input fields)
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            this.keys[e.key.toLowerCase()] = true;
        };
        this.keyUpHandler = (e) => {
            this.keys[e.key.toLowerCase()] = false;
        };

        // Mouse controls for looking
        this.mouseDownHandler = (e) => {
            if (e.target !== this.canvas) return;
            this.mouseDown = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
        };
        this.mouseUpHandler = () => {
            this.mouseDown = false;
            this.canvas.style.cursor = 'grab';
        };
        this.mouseMoveHandler = (e) => {
            if (!this.mouseDown) return;

            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;

            this.yaw -= deltaX * this.lookSpeed;
            this.pitch -= deltaY * this.lookSpeed;

            // Clamp pitch to avoid flipping
            this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));

            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        };

        // Scroll to adjust move speed
        this.wheelHandler = (e) => {
            if (e.target !== this.canvas) return;
            e.preventDefault();

            // Adjust speed with scroll
            if (e.deltaY < 0) {
                this.moveSpeed *= 1.2;
            } else {
                this.moveSpeed /= 1.2;
            }
            this.moveSpeed = Math.max(0.1, Math.min(20, this.moveSpeed));
            console.log('Move speed:', this.moveSpeed.toFixed(2));
        };

        // Add event listeners
        document.addEventListener('keydown', this.keyDownHandler);
        document.addEventListener('keyup', this.keyUpHandler);
        this.canvas.addEventListener('mousedown', this.mouseDownHandler);
        document.addEventListener('mouseup', this.mouseUpHandler);
        document.addEventListener('mousemove', this.mouseMoveHandler);
        this.canvas.addEventListener('wheel', this.wheelHandler, { passive: false });

        // Set initial cursor
        this.canvas.style.cursor = 'grab';
    }

    handleResize() {
        if (!this.renderer || !this.camera) return;

        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    updateControls(deltaTime) {
        if (!this.camera) return;

        // Calculate forward and right vectors based on yaw
        // Forward is the direction the camera is looking (into the screen = +Z when yaw=0)
        const forward = new this.THREE.Vector3(
            -Math.sin(this.yaw),
            0,
            Math.cos(this.yaw)
        );
        const right = new this.THREE.Vector3(
            Math.cos(this.yaw),
            0,
            Math.sin(this.yaw)
        );
        const up = new this.THREE.Vector3(0, -1, 0);  // Flipped Y for ml-sharp orientation

        const velocity = new this.THREE.Vector3();
        const speed = this.moveSpeed * deltaTime;

        // WASD movement - W moves toward where we're looking
        if (this.keys['w']) velocity.add(forward.clone().multiplyScalar(-speed));
        if (this.keys['s']) velocity.add(forward.clone().multiplyScalar(speed));
        if (this.keys['a']) velocity.add(right.clone().multiplyScalar(speed));
        if (this.keys['d']) velocity.add(right.clone().multiplyScalar(-speed));
        if (this.keys['q'] || this.keys[' ']) velocity.add(up.clone().multiplyScalar(-speed));
        if (this.keys['e'] || this.keys['shift']) velocity.add(up.clone().multiplyScalar(speed));

        // Apply movement
        this.camera.position.add(velocity);

        // Update camera rotation from yaw/pitch
        // Apply 180 degree rotation on Z to flip the view right-side up
        const quaternion = new this.THREE.Quaternion();
        const euler = new this.THREE.Euler(-this.pitch, this.yaw, Math.PI, 'YXZ');
        quaternion.setFromEuler(euler);
        this.camera.quaternion.copy(quaternion);
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        const now = performance.now();
        const deltaTime = (now - this.lastTime) / 1000;
        this.lastTime = now;

        // Update first-person controls
        this.updateControls(deltaTime);

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    async loadPly(url) {
        // Wait for initialization to complete first
        await this.ready;

        if (!this.SplatMesh) {
            console.error('GaussianSplatViewer: Spark.js not loaded');
            this.onError(new Error('Spark.js not loaded'));
            return;
        }

        console.log('GaussianSplatViewer: Loading PLY from:', url);
        this.onProgress(10);

        try {
            // Remove existing splat if any
            if (this.splat) {
                this.scene.remove(this.splat);
                if (this.splat.dispose) {
                    this.splat.dispose();
                }
                this.splat = null;
            }

            this.onProgress(20);

            console.log('GaussianSplatViewer: Creating SplatMesh...');

            // Create new splat mesh
            this.splat = new this.SplatMesh({
                url: url,
                onProgress: (progress) => {
                    // progress is 0-1
                    const pct = 20 + progress * 70;
                    this.onProgress(pct);
                }
            });

            console.log('GaussianSplatViewer: Waiting for loadPromise...');

            // Wait for the splat to load
            await this.splat.loadPromise;

            this.onProgress(95);

            console.log('GaussianSplatViewer: Adding splat to scene...');

            // Add to scene
            this.scene.add(this.splat);

            // Auto-center camera on the splat
            this.centerCameraOnSplat();

            this.onProgress(100);
            this.onLoad();

            console.log('GaussianSplatViewer: PLY loaded successfully!');
        } catch (error) {
            console.error('GaussianSplatViewer: Failed to load PLY:', error);
            this.onError(error);
        }
    }

    centerCameraOnSplat() {
        if (!this.splat || !this.camera) return;

        // Position camera in front of the splat, facing it
        const distance = 3.0;

        // Camera at +Z, looking toward origin (where splat is)
        this.camera.position.set(0, 0, distance);

        // Set yaw to face toward -Z (toward the splat at origin)
        this.yaw = Math.PI;
        this.pitch = 0;

        console.log('GaussianSplatViewer: Camera positioned at distance:', distance);
    }

    resetCamera() {
        this.centerCameraOnSplat();
    }

    dispose() {
        // Stop animation loop
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        // Remove event listeners
        document.removeEventListener('keydown', this.keyDownHandler);
        document.removeEventListener('keyup', this.keyUpHandler);
        this.canvas.removeEventListener('mousedown', this.mouseDownHandler);
        document.removeEventListener('mouseup', this.mouseUpHandler);
        document.removeEventListener('mousemove', this.mouseMoveHandler);
        this.canvas.removeEventListener('wheel', this.wheelHandler);

        // Remove resize observer
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }

        // Dispose splat
        if (this.splat) {
            this.scene.remove(this.splat);
            if (this.splat.dispose) {
                this.splat.dispose();
            }
            this.splat = null;
        }

        // Dispose renderer
        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }

        this.scene = null;
        this.camera = null;

        console.log('GaussianSplatViewer: Disposed');
    }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GaussianSplatViewer;
}
