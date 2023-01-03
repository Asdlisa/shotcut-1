import QtQuick
import org.shotcut.qml

Metadata {
    type: Metadata.Filter
    objectName: 'cropCircle'
    name: qsTr('Crop: Circle')
    keywords: qsTr('trim remove oval ellipse', 'search keywords for the Crop: Circle video filter') + ' crop: circle'
    mlt_service: 'qtcrop'
    qml: 'ui.qml'
    icon: 'icon.webp'

    keyframes {
        allowAnimateIn: true
        allowAnimateOut: true
        simpleProperties: ['radius']
        parameters: [
            Parameter {
                name: qsTr('Radius')
                property: 'radius'
                isCurve: true
                minimum: 0
                maximum: 1
            }
        ]
    }
}
