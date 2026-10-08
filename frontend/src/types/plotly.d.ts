declare module 'plotly.js-dist-min' {
  import Plotly from 'plotly.js'
  export default Plotly
  export * from 'plotly.js'
}

declare module 'react-plotly.js/factory' {
  import { PlotParams } from 'react-plotly.js'
  import Plotly from 'plotly.js'

  function createPlotlyComponent(plotly: typeof Plotly): React.ComponentType<PlotParams>
  export default createPlotlyComponent
}
