#include <cstdint>
#include <memory>
#include <functional>



namespace momms
{
    class packer;

    class rung
    {
    public:

        rung();

    std::size_t reuse();
    std::size_t invariant();

    template<typename scalar>
    void run(std::vector<scalar*> component_pointers,
            std::function<void(std::vector<scalar*>&&)> per_step)
    {
        auto moving_pointers = component_pointers;
        auto end_pointers = moving_pointers;
        end_pointers[component] += range;
        for(; moving_pointers[component] < end_pointers[component]; moving_pointers[component] += step)
        {
            if (pack != nullptr)
            {
                pack->pack(moving_pointers);
            }
            moving_pointers[component] += step;
            per_step(moving_pointers);
        }
    }

    private:
        std::unique_ptr<packer> pack;
        std::uint64_t range;
        std::uint64_t step;
        std::uint64_t component;
    };
}
