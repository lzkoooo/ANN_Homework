import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12
import QtCharts 2.3

import '..'

Page {
    name: 'SOM'
    ColumnLayout {
        GroupBox {
            title: 'Dataset'
            Layout.fillWidth: true
            ComboBox {
                id: datasetCombobox
                anchors.fill: parent
                model: Object.keys(somBridge.dataset_dict)
                enabled: somBridge.has_finished
                onActivated: () => {
                    somBridge.current_dataset_name = currentText
                    dataChart.updateDataset(somBridge.dataset_dict[datasetCombobox.currentText])
                }
                Component.onCompleted: () => {
                    somBridge.current_dataset_name = currentText
                    dataChart.updateDataset(somBridge.dataset_dict[datasetCombobox.currentText])
                }
            }
        }
        GroupBox {
            title: 'Settings'
            Layout.fillWidth: true
            GridLayout {
                anchors.fill: parent
                columns: 2
                Label {
                    text: 'Total Training Epoches'
                    Layout.alignment: Qt.AlignHCenter
                }
                SpinBox {
                    id: totalEpoches
                    enabled: somBridge.has_finished
                    editable: true
                    value: 10
                    to: 999999
                    onValueChanged: somBridge.total_epoches = value
                    Component.onCompleted: somBridge.total_epoches = value
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Initial Learning Rate'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: somBridge.has_finished
                    editable: true
                    value: 0.8 * 100
                    onValueChanged: somBridge.initial_learning_rate = value / 100
                    Component.onCompleted: somBridge.initial_learning_rate = value / 100
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Initial Standard Deviation'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: somBridge.has_finished
                    editable: true
                    value: 1 * 100
                    from: 0.01 * 100
                    to: 99999 * 100
                    onValueChanged: somBridge.initial_standard_deviation = value / 100
                    Component.onCompleted: somBridge.initial_standard_deviation = value / 100
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Topology Shape'
                    Layout.alignment: Qt.AlignHCenter
                }
                RowLayout {
                    Layout.fillWidth: true
                    SpinBox {
                        id: topologyRowCount
                        enabled: somBridge.has_finished
                        editable: true
                        value: 10
                        from: 1
                        to: 999
                        onValueChanged: somBridge.topology_shape = [value, topologyColumnCount.value]
                        Component.onCompleted: somBridge.topology_shape = [value, topologyColumnCount.value]
                        Layout.fillWidth: true
                    }
                    Label { text: '*' }
                    SpinBox {
                        id: topologyColumnCount
                        enabled: somBridge.has_finished
                        editable: true
                        value: 10
                        from: 1
                        to: 999
                        onValueChanged: somBridge.topology_shape = [topologyRowCount.value, value]
                        Component.onCompleted: somBridge.topology_shape = [topologyRowCount.value, value]
                        Layout.fillWidth: true
                    }
                }
                Label {
                    text: 'UI Refresh Interval'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: somBridge.has_finished
                    editable: true
                    value: 0.05 * 100
                    from: 0 * 100
                    to: 5 * 100
                    onValueChanged: somBridge.ui_refresh_interval = value / 100
                    Component.onCompleted: somBridge.ui_refresh_interval = value / 100
                    Layout.fillWidth: true
                }
            }
        }
        GroupBox {
            title: 'Information'
            Layout.fillWidth: true
            Layout.fillHeight: true
            GridLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                columns: 2
                ExecutionControls {
                    startButton.enabled: somBridge.has_finished
                    startButton.onClicked: somBridge.start_som_algorithm()
                    stopButton.enabled: !somBridge.has_finished
                    stopButton.onClicked: somBridge.stop_som_algorithm()
                    progressBar.value: (somBridge.current_iterations + 1) / (totalEpoches.value * somBridge.dataset_dict[datasetCombobox.currentText].length)
                    Layout.columnSpan: 2
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Current Training Epoch'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: currentEpoch()
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                    function currentEpoch() {
                        const epoch = Math.floor(somBridge.current_iterations / somBridge.dataset_dict[datasetCombobox.currentText].length) + 1
                        if (isNaN(epoch))
                            return 1
                        return epoch
                    }
                }
                Label {
                    text: 'Current Training Iteration'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: somBridge.current_iterations + 1
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                    onTextChanged: () => {
                        dataChart.updateNeuron()
                    }
                }
                Label {
                    text: 'Current Learning Rate'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: somBridge.current_learning_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Current Standard Deviation'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: somBridge.current_standard_deviation.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
            }
        }
    }
    ColumnLayout {
        DataChart {
            id: dataChart
            property var neuronScatter
            property var neuronTopology: []
            
            width: 700
            Layout.fillWidth: true
            Layout.fillHeight: true

            function updateNeuron() {
                updateNeuronTopology()
                updateNeuronScatter()
            }

            function updateNeuronScatter() {
                if (neuronScatter) {
                    removeSeries(neuronScatter)
                }
                neuronScatter = createHoverableScatterSeries('SOM')
                neuronScatter.color = 'black'
                for (let row of somBridge.current_synaptic_weights) {
                    for (let synaptic_weight of row) {
                        neuronScatter.append(...synaptic_weight)
                    }
                }
            }

            function updateNeuronTopology() {
                neuronTopology.forEach(line => {removeSeries(line)})
                neuronTopology = []
                createMapLines(somBridge.current_synaptic_weights)
                if (somBridge.current_synaptic_weights[0]) {
                    const transposed = somBridge.current_synaptic_weights[0].map(
                        (col, i) => somBridge.current_synaptic_weights.map(row => row[i])
                    )
                    createMapLines(transposed)
                }
            }

            function createMapLines(synaptic_weights) {
                for (let row of synaptic_weights) {
                    const line = createSeries(
                        ChartView.SeriesTypeLine, name, xAxis, yAxis
                    )
                    line.color = 'grey'
                    neuronTopology.push(line)
                    for (let synaptic_weight of row) {
                        line.append(...synaptic_weight)
                    }
                }
            }

            // XXX: sadly, QML does not have `super` access to the base class. Thus,
            // we cannot call super method in overridden methods.
            // Further details: https://bugreports.qt.io/browse/QTBUG-25942
            function clear() {
                removeAllSeries()
                scatterSeriesMap = {}
                neuronTopology = []
            }
        }
    }
}