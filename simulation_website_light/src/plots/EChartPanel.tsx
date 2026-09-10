import { BarChart, HeatmapChart, LineChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import type { EChartsOption } from 'echarts'
import type { EChartsType } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'

echarts.use([
  LineChart,
  BarChart,
  HeatmapChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  VisualMapComponent,
  MarkLineComponent,
  CanvasRenderer,
])

interface EChartPanelProps {
  option: EChartsOption
  onTimeSelect?: (time: number) => void
}

export function EChartPanel({ option, onTimeSelect }: EChartPanelProps) {
  const container = useRef<HTMLDivElement>(null)
  const chart = useRef<EChartsType | null>(null)

  useEffect(() => {
    if (!container.current) return
    const instance = echarts.init(container.current, undefined, {
      renderer: 'canvas',
      devicePixelRatio: Math.min(window.devicePixelRatio, 2),
    })
    chart.current = instance
    const resizeObserver = new ResizeObserver(() => instance.resize())
    resizeObserver.observe(container.current)
    return () => {
      resizeObserver.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [])

  useEffect(() => {
    chart.current?.setOption(option, true)
  }, [option])

  useEffect(() => {
    const instance = chart.current
    if (!instance || !onTimeSelect) return
    const handler = (params: any) => {
      const value = params?.value
      if (Array.isArray(value) && Number.isFinite(Number(value[0]))) {
        onTimeSelect(Number(value[0]))
      }
    }
    instance.on('click', handler)
    return () => {
      instance.off('click', handler)
    }
  }, [onTimeSelect])

  return <div ref={container} className="echart" role="img" aria-label="Interactive simulation chart" />
}
