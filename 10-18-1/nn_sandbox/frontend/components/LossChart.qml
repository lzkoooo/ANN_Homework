import QtQuick 2.15
import QtQuick.Controls 2.15
import QtDataVisualization 1.2

Surface3D {
    id: surface3D
    anchors.fill: parent
    property var ori_pict: Surface3DSeries{
        id: ori_pict
        flatShadingEnabled: true
        drawMode: Surface3DSeries.DrawSurface
    }

    property var current_loss: Surface3DSeries{
        id: current_loss
        flatShadingEnabled: true
        drawMode: Surface3DSeries.DrawSurface
    }

    antialiasing: true

    ValueAxis{
        id: xAxis
        titleText: 'W 1'
        min: -10
        max: 10
    }
    ValueAxis{
        id: yAxis
        titleText: 'W 2'
        min: -10
        max: 10
    }
    ValueAxis{
        id: zAxis
        titleText: 'Loss'
    }

    function updateAxes(point) {
        xAxis.max = Math.max(xAxis.max, point.x)
        xAxis.min = Math.min(xAxis.min, point.x)
        yAxis.max = Math.max(yAxis.max, point.y)
        yAxis.min = Math.min(yAxis.min, point.y)
    }

    function reset() {
        current_loss = createSeries(
            Surface3D.SeriesTypeLine, 'loss', xAxis, yAxis, zAxis
        )
        current_loss.pointAdded.connect((index) => {
            updateAxes(current_loss.at(index))
        })
        xAxis.max = 1
        xAxis.min = 0
        yAxis.max = 1
        yAxis.min = 0
    }

    Component.onCompleted: reset()
}

    // 添加图形代理和数据项
    Surface3DSeries {
        SurfaceDataProxy {
            // 创建一个简单的网格，设置顶点数据
            ItemModelSurfaceDataProxy {
                rowCount: 100
                columnCount: 100
                sourceItemModel: model
            }
        }

        // 设置样式
        flatShadingEnabled: true
        drawMode: Surface3DSeries.DrawSurface
    }
}

    // 模拟数据模型
    ListModel {
        id: model
        Component.onCompleted: {
            for (let i = 0; i < 50; i++) {
                for (let j = 0; j < 50; j++) {
                    model.append({"x": i, "z": j, "y": Math.sin(i / 5.0) * Math.cos(j / 5.0) * 5});
                }
            }
        }
    }

    // Timer用于动态更新数据
    Timer {
        interval: 100  // 每100毫秒更新一次
        running: true
        repeat: true
        onTriggered: {
            // 更新模型中的数据
            for (let i = 0; i < 50; i++) {
                for (let j = 0; j < 50; j++) {
                    // 动态改变y值
                    let index = i * 50 + j;
                    let newY = Math.sin(i / 5.0 + Date.now() / 1000.0) * Math.cos(j / 5.0) * 5;
                    model.set(index, {"x": i, "z": j, "y": newY});
                }
            }
        }
    }
