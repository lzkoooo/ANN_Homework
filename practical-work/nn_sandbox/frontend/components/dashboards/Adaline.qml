import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.5
import QtQuick.Layouts 1.12

import '..'

Page {
    name: 'Adaline'
    ColumnLayout {
        GroupBox {
            title: 'Dataset'
            Layout.fillWidth: true
            ComboBox {
                id: datasetCombobox
                anchors.fill: parent
                model: Object.keys(adalineBridge.dataset_dict)
                enabled: adalineBridge.has_finished
                onActivated: () => {
                    adalineBridge.current_dataset_name = currentText
                    dataChart.updateDataset(adalineBridge.dataset_dict[datasetCombobox.currentText])
                }
                Component.onCompleted: () => {
                    adalineBridge.current_dataset_name = currentText
                    dataChart.updateDataset(adalineBridge.dataset_dict[datasetCombobox.currentText])
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
                    enabled: adalineBridge.has_finished
                    editable: true
                    value: 10
                    to: 999999
                    onValueChanged: adalineBridge.total_epoches = value
                    Component.onCompleted: adalineBridge.total_epoches = value
                    Layout.fillWidth: true
                }
                CheckBox {
                    id: mostCorrectRateCheckBox
                    enabled: adalineBridge.has_finished
                    text: 'Most Correct Rate'
                    checked: true
                    onCheckedChanged: adalineBridge.most_correct_rate_checkbox = checked
                    Component.onCompleted: adalineBridge.most_correct_rate_checkbox = checked
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: mostCorrectRateCheckBox.checked && adalineBridge.has_finished
                    editable: true
                    value: 1.00 * 100
                    onValueChanged: adalineBridge.most_correct_rate = value / 100
                    Component.onCompleted: adalineBridge.most_correct_rate = value / 100
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Initial Learning Rate'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: adalineBridge.has_finished
                    editable: true
                    value: 0.8 * 100
                    onValueChanged: adalineBridge.initial_learning_rate = value / 100
                    Component.onCompleted: adalineBridge.initial_learning_rate = value / 100
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Test-Train Ratio'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: adalineBridge.has_finished
                    editable: true
                    value: 0.3 * 100
                    from: 30
                    to: 90
                    onValueChanged: adalineBridge.test_ratio = value / 100
                    Component.onCompleted: adalineBridge.test_ratio = value / 100
                    Layout.fillWidth: true
                }
                Label {
                    text: 'UI Refresh Interval'
                    Layout.alignment: Qt.AlignHCenter
                }
                DoubleSpinBox {
                    enabled: adalineBridge.has_finished
                    editable: true
                    value: 0.0 * 100
                    from: 0 * 100
                    to: 5 * 100
                    onValueChanged: adalineBridge.ui_refresh_interval = value / 100
                    Component.onCompleted: adalineBridge.ui_refresh_interval = value / 100
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
                    applyButton1.visible: true
                    applyButton1.text: "9月27日第三题"
                    applyButton2.visible: true
                    applyButton2.text: "9月27日第四题"

                    startButton.enabled: adalineBridge.has_finished
                    startButton.onClicked: () => {
                        adalineBridge.start_adaline_algorithm()
                        dataChart.clear()
                        dataChart.updateTrainingDataset(adalineBridge.training_dataset)
                        dataChart.updateTestingDataset(adalineBridge.testing_dataset)
                        rateChart.reset()
                        if (datasetCombobox.currentText === 'Adaline_9月27日第3题train')
                        {applyButton1.enabled = true;}
                        else if (datasetCombobox.currentText === 'Adaline_9月27日第4题train')
                        {applyButton2.enabled = true;}
                    }
                    stopButton.enabled: !adalineBridge.has_finished
                    stopButton.onClicked: {
                        adalineBridge.stop_adaline_algorithm()
                    }
                    applyButton1.onClicked:{
                        adalineBridge.apply_topic = 3
                        adalineBridge.apply_adaline_algorithm()
                    }
                    applyButton2.onClicked:{
                        adalineBridge.apply_topic = 4
                        adalineBridge.apply_adaline_algorithm()
                    }
                    progressBar.value: (adalineBridge.current_iterations + 1) / (totalEpoches.value * adalineBridge.training_dataset.length)
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
                        const epoch = Math.floor(adalineBridge.current_iterations / adalineBridge.training_dataset.length) + 1
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
                    text: adalineBridge.current_iterations + 1
                    horizontalAlignment: Text.AlignHCenter
                    onTextChanged: () => {
                        rateChart.bestCorrectRate.append(
                            adalineBridge.current_iterations + 1,
                            adalineBridge.best_correct_rate
                        )
                        rateChart.trainingCorrectRate.append(
                            adalineBridge.current_iterations + 1,
                            adalineBridge.current_correct_rate
                        )
                        rateChart.testingCorrectRate.append(
                            adalineBridge.current_iterations + 1,
                            adalineBridge.test_correct_rate
                        )
                    }
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Current Learning Rate'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: adalineBridge.current_learning_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Best Training Correct Rate'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: adalineBridge.best_correct_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Current Training Correct Rate'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: adalineBridge.current_correct_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
                Label {
                    text: 'Current Testing Correct Rate'
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    text: adalineBridge.test_correct_rate.toFixed(toFixedValue)
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true
                }
            }
        }
    }
    ColumnLayout {
        DataChart {
            id: dataChart
            width: 700
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
        RateChart {
            id: rateChart
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}