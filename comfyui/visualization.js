// Thanks to MrForExample ComfyUI-3D-Pack for the base code
// https://github.com/MrForExample/ComfyUI-3D-Pack/blob/main/web/visualization.js

import { app } from "/scripts/app.js"

class Visualizer {
    constructor(node, container) {
        this.node = node;
        this.iframe = document.createElement('iframe');
        Object.assign(this.iframe, {
            scrolling: "no",
            overflow: "hidden",
        });
        this.iframe.src = "/extensions/stable-fast-3d/web/visualizer.html";
        container.appendChild(this.iframe);
    }

    update(b64_glb) {
        const iframeDocument = this.iframe.contentWindow.document;
        const previewScript = iframeDocument.getElementById('visualizer');
        previewScript.setAttribute("b64_glb", b64_glb);
        previewScript.setAttribute("timestamp", Date.now().toString());
    }

    remove() {
        this.container.remove();
    }
}

function createWidget(node, app) {
    const widget = {
        type: "StableFast3DViewer",
        name: "preview",
        callback: () => { },
        draw: function (ctx, node, widgetWidth, widgetY, widgetHeight) {
            const margin = 10;
            const top_offset = 5;
            const visible = app.canvas.ds.scale > 0.5;
            const w = widgetWidth - margin * 4;
            const clientRectBound = ctx.canvas.getBoundingClientRect();
            const transform = new DOMMatrix()
                .scaleSelf(
                    clientRectBound.width / ctx.canvas.width,
                    clientRectBound.height / ctx.canvas.height
                )
                .multiplySelf(ctx.getTransform())
                .translateSelf(margin, margin + widgetY);

            Object.assign(this.visualizer.style, {
                left: `${transform.a * margin + transform.e}px`,
                top: `${transform.d + transform.f + top_offset}px`,
                width: `${(w * transform.a)}px`,
                height: `${(w * transform.d - widgetHeight - (margin * 15) * transform.d)}px`,
                position: "absolute",
                overflow: "hidden",
                zIndex: app.graph._nodes.indexOf(node),
            });

            Object.assign(this.visualizer.children[0].style, {
                transformOrigin: "50% 50%",
                width: '100%',
                height: '100%',
                border: '0 none',
            });

            this.visualizer.hidden = !visible;
        },
    };

    const container = document.createElement('div');

    node.visualizer = new Visualizer(node, container);
    widget.visualizer = container;
    widget.parent = node;
    document.body.appendChild(widget.visualizer);
    node.addCustomWidget(widget);

    node.onDrawBackground = (ctx) => {
        node.visualizer.iframe.hidden = this.flags.collapsed;
    };

    node.onRemoved = () => {
        for (let w in node.widgets) {
            if (node.widgets[w].visualizer) {
                node.widgets[w].visualizer.remove();
            }
        }
    };

    node.onResize = () => {
        let [w, h] = this.size;
        if (w <= 600) w = 600;
        if (h <= 500) h = 500;
        if (w > 600) {
            h = w - 100;
        }
        this.size = [w, h];
    };

    node.updateParameters = (b64_glb) => {
        node.visualizer.update(b64_glb);
    };

    return { widget: widget }
}


function registerVisualizer(nodeType, nodeData) {
    if (nodeData.name !== "StableFast3DSave" && nodeData.name !== "StableFast3DPreview")
        return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = async function () {
        const result = originalOnNodeCreated?.apply(this, arguments);
        await createWidget.apply(this, [this, app]);
        this.setSize([512, 512]);
        return result;
    };

    nodeType.prototype.onExecuted = function (message) {
        if (message?.glbs?.[0]) {
            this.updateParameters(message.glbs[0]);
        }
    };
}

app.registerExtension({
    name: "StableFast3D.Visualizer",
    async init(app) { },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        registerVisualizer(nodeType, nodeData);
    },
});
