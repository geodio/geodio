//    Geodio Neural Network Framework - A lightweight neural network library focused on custom architecture design and optimization.
//    Copyright (C) Copyright (C) 2024 Geodio (created by Rareș Polenciuc)
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#include "CPUBackend.h"
#include "Backend.h"
#include <memory>

#ifndef GEODIO_BACKENDMANAGER_H
#define GEODIO_BACKENDMANAGER_H

namespace dio{
    template<typename T>
    class BackendManager {
    public:
        static std::shared_ptr<Backend<T>> get_backend() {
            static std::shared_ptr<Backend<T>> instance = std::make_shared<CPUBackend<T>>();
            return instance;
        }
    };
} // namespace dio




#endif //GEODIO_BACKENDMANAGER_H
