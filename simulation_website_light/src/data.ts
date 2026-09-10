import type { AppData, BaselineData, HillData, ModelData } from './simulation/model-types'

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}${path}`)
  if (!response.ok) {
    throw new Error(`Could not load ${path}: ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

export async function loadAppData(): Promise<AppData> {
  const [model, baseline, hill] = await Promise.all([
    fetchJson<ModelData>('data/model.json'),
    fetchJson<BaselineData>('data/baseline-flight.json'),
    fetchJson<HillData>('data/hill-profile.json'),
  ])
  return { model, baseline, hill }
}

